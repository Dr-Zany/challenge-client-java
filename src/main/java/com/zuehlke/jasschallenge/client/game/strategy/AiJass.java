package com.zuehlke.jasschallenge.client.game.strategy;

import ai.onnxruntime.*; // Import necessary classes
import com.zuehlke.jasschallenge.client.game.GameSession;
import com.zuehlke.jasschallenge.client.game.Move;
import com.zuehlke.jasschallenge.client.game.Round;
import com.zuehlke.jasschallenge.game.cards.Card;
import com.zuehlke.jasschallenge.game.mode.Mode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.TimeUnit;
import java.nio.LongBuffer;
import java.util.*;


public class AiJass implements JassStrategy {

    private static final Logger log = LoggerFactory.getLogger(AiJass.class);
    OrtEnvironment o_environment;
    OrtSession.SessionOptions o_options;
    OrtSession o_sessionPlay;
    OrtSession o_sessionTrump;
    OrtSession o_sessionTime;

    int moveCount = 0;
    List<Card> history = new Vector<>();
    List<Card> onTable = new Vector<>();

    public AiJass(String playPath, String trumpPath, String timePath) throws OrtException {
        o_environment = OrtEnvironment.getEnvironment();
        o_options = new OrtSession.SessionOptions();

        o_sessionPlay = o_environment.createSession(playPath, o_options);
        o_sessionTrump = o_environment.createSession(trumpPath, o_options);
        o_sessionTime = o_environment.createSession(timePath, o_options);
    }


    private OnnxTensor creatState(List<Card> hand, Mode mode) throws OrtException {
        long[] stateIndices = new long[72];
        int i = 0;

        for(int y = 0; y < history.size(); y++) {
            stateIndices[y + i] = history.get(y).ordinal() + 1;
        }
        i += history.size();
        while(i < 32) {stateIndices[i] = 0; i++;}

        for (int y = 0; y < onTable.size(); y++) {
            stateIndices[y + i] = onTable.get(y).ordinal() + 1;
        }
        i += onTable.size();
        while(i < 35) {stateIndices[i] = 0; i++;}

        for(int y = 0; y < hand.size(); y++) {
            stateIndices[y + i] = hand.get(y).ordinal() + 1;
        }
        i += hand.size();
        while(i < 44) {stateIndices[i] = 0; i++;}

        while(i < stateIndices.length) {stateIndices[i] = 0; i++;}

        switch (mode.getTrumpfName()){
            case OBEABE:
                stateIndices[71] = 6;
                break;

            case UNDEUFE:
                stateIndices[71] = 5;
                break;

            case TRUMPF:
                stateIndices[71] = mode.getTrumpfColor().ordinal() + 1;
                break;

            case SCHIEBE:
                stateIndices[71] = 0;
        }

        return OnnxTensor.createTensor(
                o_environment,
                LongBuffer.wrap(stateIndices),
                new long[]{1, 72}
        );
    }

    private OnnxTensor creatTrumpState(List<Card> hand, boolean isGschobe) throws OrtException {
        long[] stateIndices = new long[10];
        int i = 0;
        for(Card card : hand) {
            stateIndices[i] = card.ordinal() + 1;
        }
        stateIndices[9] = isGschobe ? 1 : 0;

        return OnnxTensor.createTensor(
                o_environment,
                LongBuffer.wrap(stateIndices),
                new long[]{1, 10}
        );
    }

    private long getTime(List<Card> hand, Mode mode){
        try {
            OnnxTensor state = creatState(hand, mode);
            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("state", state);

            try(OrtSession.Result output = o_sessionTime.run(input)) {
                OnnxTensor action = (OnnxTensor) output.get("action").get();
                long time = (long) ((float[][]) action.getValue())[0][0];
                log.info("Time for action: {}", time);
                return time;
            }

        } catch (OrtException e) {
            log.error(e.getMessage());
        }
        return 10;
    }

    @Override
    public Mode chooseTrumpf(Set<Card> availableCards, GameSession session, boolean isGschobe) {
        List<Card> hand = new ArrayList<>(availableCards);
        long time = getTime(hand, Mode.shift()); // shift is mapped to no trump
        try {
            OnnxTensor state = creatTrumpState(hand, isGschobe);
            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("state", state);

            try(OrtSession.Result output = o_sessionTrump.run(input)) {
                OnnxTensor action = (OnnxTensor) output.get("action").get();
                float[] actionProbabilities = action.getFloatBuffer().array();

                List<Map.Entry<Integer, Float>> prob = new ArrayList<>();
                for(int i = 0; i < actionProbabilities.length; i++ ) {
                    prob.add(new AbstractMap.SimpleEntry<>(i, actionProbabilities[i]));
                }

                prob.sort(Map.Entry.comparingByValue());

                for(Map.Entry<Integer, Float> entry : prob) {
                    log.info("Probability: {} {}", entry.getKey() ,entry.getValue());
                }

                for(int i = prob.size() - 1; i >= 0; i--) {
                    Map.Entry<Integer, Float> probEntry = prob.get(i);
                    if(probEntry.getKey() == 0 && !isGschobe){
                        log.info("Trumpf gschobe: {}", probEntry.getValue());
                        return Mode.shift();
                    }

                    else{
                        Mode mode = Mode.standardModes().get(probEntry.getKey() - 1);
                        log.info("Trumpf: {}", mode);
                        return mode;
                    }

                }
            }

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return null;
    }

    @Override
    public Card chooseCard(Set<Card> availableCards, GameSession session) {
        List<Card> cards = new ArrayList<>(availableCards);
        Round round = session.getCurrentRound();
        long time = getTime(cards, round.getMode());
        try {
            OnnxTensor state = creatState(cards, round.getMode());

            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("state", state);

            try(OrtSession.Result output = o_sessionPlay.run(input)) {
                OnnxTensor action = (OnnxTensor)output.get("action").get();
                float[] actionProbabilities = action.getFloatBuffer().array();

                List<Map.Entry<Integer, Float>> prob = new ArrayList<>();
                for(int i = 0; i < actionProbabilities.length; i++ ) {
                    prob.add(new AbstractMap.SimpleEntry<>(i, actionProbabilities[i]));
                }

                prob.sort(Map.Entry.comparingByValue());

                for(Map.Entry<Integer, Float> probEntry : prob) {
                    if(probEntry.getKey() < cards.size()) {
                        log.info("{}:{}", cards.get(probEntry.getKey()).toString(), probEntry.getValue());
                    }
                    else {
                        log.info("Empty: {}", probEntry.getValue());
                    }
                }
                TimeUnit.SECONDS.sleep(time);
                for(int i = prob.size() - 1; i >= 0; i--) {
                    if(prob.get(i).getKey() >= cards.size()) {
                        log.info("trying to play empty card");
                        continue;
                    }
                    if(round.isLegal(cards.get(prob.get(i).getKey()), cards)) {
                        return cards.get(prob.get(i).getKey());
                    }
                    else{
                        log.info("trying to play illegal card {}", cards.get(prob.get(i).getKey()));
                    }
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return null;
    }

    @Override
    public void onMoveMade(Move move, GameSession session) {
        moveCount++;
        onTable.add(move.getPlayedCard());

        if(moveCount % 4 == 0){
            history.addAll(onTable);
            onTable.clear();
        }

        if (moveCount % 36 == 0){
            history.clear();
            onTable.clear();
            moveCount = 0;
        }


    }






}
