import com.sun.net.httpserver.HttpServer;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.JSONOutputter;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Properties;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Minimal pinned-library transport: never binds a wildcard address. */
public final class LoopbackCoreNLP {
    public static void main(String[] args) throws Exception {
        Properties props = new Properties();
        props.setProperty("annotators", StanfordCoreNLP.ensurePrerequisiteAnnotators(
            new String[]{"tokenize", "ssplit", "lemma"}, props));
        props.setProperty("outputFormat", "json");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        HttpServer server = HttpServer.create(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), 0), 8);
        ThreadPoolExecutor worker = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.SECONDS,
            new ArrayBlockingQueue<Runnable>(8), new ThreadPoolExecutor.AbortPolicy());
        server.setExecutor(worker);
        server.createContext("/", exchange -> {
            int status = 200;
            byte[] result;
            try {
                String path = exchange.getRequestURI().getPath();
                if (path.equals("/ready") && exchange.getRequestMethod().equals("GET")) {
                    result = "ready".getBytes(StandardCharsets.UTF_8);
                } else if (path.equals("/") && exchange.getRequestMethod().equals("POST")) {
                    byte[] body = exchange.getRequestBody().readNBytes(1048577);
                    if (body.length > 1048576) {
                        status = 413;
                        result = new byte[0];
                    } else {
                        Annotation annotation = new Annotation(new String(body, StandardCharsets.UTF_8));
                        pipeline.annotate(annotation);
                        result = JSONOutputter.jsonPrint(annotation).getBytes(StandardCharsets.UTF_8);
                    }
                } else {
                    status = 404;
                    result = new byte[0];
                }
            } catch (Exception error) {
                status = 500;
                result = "annotation failed".getBytes(StandardCharsets.UTF_8);
            }
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
            exchange.sendResponseHeaders(status, result.length);
            exchange.getResponseBody().write(result);
            exchange.close();
        });
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop(0);
            worker.shutdownNow();
        }));
        server.start();
        System.out.println("DAIL_READY http://127.0.0.1:" + server.getAddress().getPort());
        System.out.flush();
    }
}
