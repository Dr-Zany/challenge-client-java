package com.zuehlke.jasschallenge.client.game.strategy;

import ai.onnxruntime.*; // Import necessary classes
import com.zuehlke.jasschallenge.client.game.GameSession;
import com.zuehlke.jasschallenge.client.game.Move;
import com.zuehlke.jasschallenge.game.cards.Card;
import com.zuehlke.jasschallenge.game.mode.Mode;

import java.util.*;

public class AiJass implements JassStrategy {

    OrtEnvironment o_environment;
    OrtSession.SessionOptions o_options;
    OrtSession o_session;

    int moveCount = 0;
    Vector<Card> history = new Vector<>();
    Vector<Card> ontable = new Vector<>();

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

    private float[] convertCard(Card card) {
        float[] cardVec = new float[13];
        for (float i : cardVec)
            i = 0;

        switch (card.getColor()){
            case SPADES:
                cardVec[12] = 1;
                break;
            case HEARTS:
                cardVec[11] = 1;
                break;
            case DIAMONDS:
                cardVec[10] = 1;
                break;
            case CLUBS:
                cardVec[9] = 1;
                break;
        }

        cardVec[card.getValue().getRank() - 1] = 1;

        return cardVec;
    }

    private OnnxTensor creatState(Vector<Card> hand, Mode mode) throws OrtException {
        float[] state = new float[929];
        int i = 0;
        for (Card card : history) {
            float[] cardVec = convertCard(card);
            copyArrayTo(cardVec, state, i);
            i += 13;
        }

        for(Card card : ontable) {
            float[] cardVec = convertCard(card);
            copyArrayTo(cardVec, state, i);
            i += 13;
        }

        for(Card card : hand) {
            float[] cardVec = convertCard(card);
            copyArrayTo(cardVec, state, i);
        }

        // fill next (hand.size() - 9) * 13 with Zeros
        for(int y = 0; y < (hand.size() - 9) * 13; y++)
            state[i + y] = 0;

        i += (hand.size() - 9) * 13;

        // fill rest with Zeros
        for(int y = i; y < state.length ; y++)
            state[y] = 0;

        switch (mode.getTrumpfName()){
            case OBEABE:
                state[state.length - 6] = 1;
                break;

            case UNDEUFE:
                state[state.length - 5] = 1;
                break;

            case TRUMPF:
                switch (mode.getTrumpfColor()){
                    case SPADES:
                        state[state.length - 4] = 1;
                        break;

                    case HEARTS:
                        state[state.length - 3] = 1;
                        break;

                    case DIAMONDS:
                        state[state.length - 2] = 1;
                        break;

                    case CLUBS:
                        state[state.length - 1] = 1;
                        break;
                }
        }

        long[] inputShape = {1, state.length};
        // Create the input tensor (requires multi-dimensional array)
        float[][] inputData = {state}; // Wrap the flat array in another array

        return OnnxTensor.createTensor(o_environment, inputData);
    }

    @Override
    public Mode chooseTrumpf(Set<Card> availableCards, GameSession session, boolean isGschobe) {
        return null;
    }

    @Override
    public Card chooseCard(Set<Card> availableCards, GameSession session) {
        List<Card> cards = availableCards.stream().toList();
        try {
            OnnxTensor state = creatState((Vector<Card>) cards, session.getCurrentGame().getCurrentRound().getMode());
            String inputName = o_session.getInputNames().iterator().next();
            OrtSession.Result results = o_session.run(Collections.singletonMap(inputName, state));

            String outputName = o_session.getOutputNames().iterator().next();
            Optional<OnnxValue> outputValue = results.get(outputName);

            if(outputValue.isEmpty())
                throw new IllegalArgumentException("Output name " + outputName + " not found");

            float[] values = ((float[][])outputValue.get().getValue())[0];

            int predict = 0;
            for (int i = 0; i < values.length; i++) {
                if(values[i] > values[predict])
                    predict = i;
            }

            return cards.get(predict);

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
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
