package com.zuehlke.jasschallenge.client.game.strategy;

import ai.onnxruntime.*; // Import necessary classes
import com.zuehlke.jasschallenge.client.game.GameSession;
import com.zuehlke.jasschallenge.client.game.Move;
import com.zuehlke.jasschallenge.client.game.Round;
import com.zuehlke.jasschallenge.game.cards.Card;
import com.zuehlke.jasschallenge.game.mode.Mode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.*;


public class AiJass implements JassStrategy {

    private static final Logger log = LoggerFactory.getLogger(AiJass.class);
    OrtEnvironment o_environment;
    OrtSession.SessionOptions o_options;
    OrtSession o_session;

    int moveCount = 0;
    List<Card> history = new Vector<>();
    List<Card> ontable = new Vector<>();

    public AiJass(String path) throws OrtException {
        o_environment = OrtEnvironment.getEnvironment();
        o_options = new OrtSession.SessionOptions();

        o_session = o_environment.createSession(path, o_options);
    }

    private void copyArrayTo(float[] src, float[] dest, int index) {
        if(src.length + index >= dest.length)
            throw new ArrayIndexOutOfBoundsException();
        System.arraycopy(src, 0, dest, index, src.length);
    }


    private OnnxTensor creatState(List<Card> hand) throws OrtException {
        long[] stateIndices = new long[71];
        int i = 0;

        for(int y = 0; y < history.size(); y++) {
            stateIndices[y + i] = history.get(y).ordinal() + 1;
        }
        i += history.size();
        while(i < 32) {stateIndices[i] = 0; i++;}

        for (int y = 0; y < ontable.size(); y++) {
            stateIndices[y + i] = ontable.get(y).ordinal() + 1;
        }
        i += ontable.size();
        while(i < 35) {stateIndices[i] = 0; i++;}

        for(int y = 0; y < hand.size(); y++) {
            stateIndices[y + i] = hand.get(y).ordinal() + 1;
        }
        i += hand.size();
        while(i < 44) {stateIndices[i] = 0; i++;}

        while(i < stateIndices.length) {stateIndices[i] = 0; i++;}

        OnnxTensor stateIdxTensor = OnnxTensor.createTensor(
                o_environment,
                LongBuffer.wrap(stateIndices),
                new long[]{1, 71}
        );

        return stateIdxTensor;
    }

    private OnnxTensor creatTrumpOneHot(Mode mode) throws OrtException {
        float[] trump = new float[7];

        switch (mode.getTrumpfName()){
            case OBEABE:
                trump[6] = 1;
                break;
            case UNDEUFE:
                trump[5] = 1;
                break;
            case TRUMPF:
                trump[mode.getTrumpfColor().ordinal() + 1] = 1;
                break;
            case SCHIEBE:
                break;
        }

        OnnxTensor trumpTensor = OnnxTensor.createTensor(
                o_environment,
                FloatBuffer.wrap(trump),
                new long[]{1, 7}
        );

        return trumpTensor;
    }

    @Override
    public Mode chooseTrumpf(Set<Card> availableCards, GameSession session, boolean isGschobe) {
        return null;
    }

    @Override
    public Card chooseCard(Set<Card> availableCards, GameSession session) {
        List<Card> cards = new ArrayList<>(availableCards);
        Round round = session.getCurrentRound();
        try {
            OnnxTensor state = creatState(cards);
            OnnxTensor trump = creatTrumpOneHot(round.getMode());

            Map<String, OnnxTensor> input = new HashMap<>();
            input.put("state_idx", state);
            input.put("trump_onehot", trump);

            try(OrtSession.Result output = o_session.run(input)) {
                OnnxTensor policyLogProbs = (OnnxTensor)output.get("policy_log_probs").get();
                float[] logProbabilities = policyLogProbs.getFloatBuffer().array();

                List<Map.Entry<Integer, Float>> prob = new ArrayList<>();
                for(int i = 0; i < logProbabilities.length; i++ ) {
                    prob.add(new AbstractMap.SimpleEntry<>(i, logProbabilities[i]));
                }

                prob.sort(Map.Entry.comparingByValue());

                for(Map.Entry<Integer, Float> probEntry : prob) {
                    if(probEntry.getKey() < cards.size()) {
                        log.info(cards.get(probEntry.getKey()).toString() + ":" + probEntry.getValue());
                    }
                    else {
                        log.info("Empty: " + probEntry.getValue());
                    }
                }

                for(int i = prob.size() - 1; i >= 0; i--) {
                    if(prob.get(i).getKey() >= cards.size()) {
                        log.info("trying to play empty card");
                        continue;
                    }
                    if(round.isLegal(cards.get(prob.get(i).getKey())))
                       return cards.get(prob.get(i).getKey());
                }
            }

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return null;
    }

    @Override
    public void onMoveMade(Move move, GameSession session) {
        moveCount++;
        ontable.add(move.getPlayedCard());

        if(moveCount % 4 == 0){
            history.addAll(ontable);
            ontable.clear();
        }

        if (moveCount % 36 == 0){
            history.clear();
            ontable.clear();
            moveCount = 0;
        }


    }






}
