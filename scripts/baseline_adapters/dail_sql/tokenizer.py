"""Local-only CoreNLP 3.9.2 HTTP client and owned loopback service lifecycle."""

import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import httpx


class CoreNLPTokenizer:
    def __init__(self, url: str):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password:
            raise ValueError("CoreNLP endpoint must be HTTP on literal 127.0.0.1")
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=120, trust_env=False)

    def tokenize_for_copying(self, text: str) -> tuple[list[str], list[str]]:
        response = self.client.post(self.url, params={"properties": json.dumps({
            "annotators": "tokenize,ssplit,lemma", "outputFormat": "json"})},
            content=text.encode("utf-8"), headers={"Content-Type": "text/plain; charset=utf-8"})
        response.raise_for_status()
        data = response.json()
        lemmas, original = [], []
        for sentence in data["sentences"]:
            for token in sentence["tokens"]:
                lemmas.append(token["lemma"].lower())
                original.append(token["originalText"].lower())
        return lemmas, original

    def tokenize(self, text: str) -> list[str]:
        return self.tokenize_for_copying(text)[0]

    def close(self):
        self.client.close()


class LocalCoreNLP:
    """One bounded worker; ephemeral OS-selected loopback port; clean teardown."""
    def __init__(self, resources: dict, output: Path, *, root: Path):
        self.resources, self.output, self.root = resources, Path(output), Path(root)
        self.process = self.log = self.client = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        java_home = Path(self.resources["java"]["home"])
        jars = Path(self.resources["corenlp"]["directory"])
        if not jars.is_absolute():
            jars = self.root / jars
        source = Path(__file__).with_name("LoopbackCoreNLP.java")
        classpath = str(jars / "*")
        subprocess.run([str(java_home / "bin/javac"), "-cp", classpath, "-d", str(self.output), str(source)],
                       check=True, capture_output=True, timeout=60)
        self.log = (self.output / "corenlp.log").open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen([str(java_home / "bin/java"), "-Xmx2g", "-cp",
                classpath + ":" + str(self.output), "LoopbackCoreNLP"], stdout=self.log, stderr=self.log)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Local CoreNLP exited during startup; inspect corenlp.log")
                lines = (self.output / "corenlp.log").read_text().splitlines()
                ready = [line for line in lines if line.startswith("DAIL_READY ")]
                if ready:
                    self.client = CoreNLPTokenizer(ready[-1].split(" ", 1)[1])
                    response = self.client.client.get(self.client.url + "/ready")
                    response.raise_for_status()
                    return self.client
                time.sleep(0.1)
            raise TimeoutError("Local CoreNLP startup timed out")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.client:
            self.client.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log:
            self.log.close()
