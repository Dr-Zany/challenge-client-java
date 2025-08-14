package com.zuehlke.jasschallenge;

import com.zuehlke.jasschallenge.client.RemoteGame;
import com.zuehlke.jasschallenge.client.game.Player;
import com.zuehlke.jasschallenge.client.game.strategy.AiJass;
import com.zuehlke.jasschallenge.client.game.strategy.JassStrategy;
import com.zuehlke.jasschallenge.client.game.strategy.RandomJassStrategy;
import com.zuehlke.jasschallenge.messages.type.SessionType;
import java.util.Arrays;

/**
 * Starts one bot in tournament mode. Add your own strategy to compete in the Jass Challenge Tournament 2017!
 * <br><br>
 * To start from CLI use
 * <pre>
 *     gradlew run [websocketUrl]
 * </pre>
 */
public class Application {
    //CHALLENGE2017: Set your bot name
    private static final String BOT_NAME = "Bob";
    //CHALLENGE2017: Set your own strategy

    private static final String LOCAL_URL = "ws://127.0.0.1:3000";

    public static void main(String[] args) throws Exception {
        String websocketUrl = System.getenv().getOrDefault("URL", LOCAL_URL);
        String name = System.getenv().getOrDefault("NAME", BOT_NAME);
        if(System.getenv("MODEL_PATH") == null) {
            throw new IllegalArgumentException("MODEL_PATH env var is required");
        }
        String modelPlayPath = System.getenv("MODEL_PATH") + "jass_play_dnn.onnx"  ;
        String modelTrumpPath = System.getenv("MODEL_PATH") + "jass_trump_dnn.onnx";


        JassStrategy strategy = new AiJass(modelPlayPath, modelTrumpPath);
        Player myLocalPlayer = new Player(name, strategy);

        System.out.println("Connecting... Server socket URL: " + websocketUrl);
        startGame(websocketUrl, myLocalPlayer, SessionType.TOURNAMENT);
    }


    private static String parseWebsocketUrlOrDefault(String[] args) {
        if (args.length > 0) {
            System.out.println("Arguments: " + Arrays.toString(args));
            return args[0];
        }
        return LOCAL_URL;
    }

    private static void startGame(String targetUrl, Player myLocalPlayer, SessionType sessionType) throws Exception {
        RemoteGame remoteGame = new RemoteGame(targetUrl, myLocalPlayer, sessionType);
        remoteGame.start();
    }
}
