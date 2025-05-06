package com.zuehlke.jasschallenge.game.cards;

public enum Color {
    DIAMONDS("(D)"),
    CLUBS("(C)"),
    HEARTS("(H)"),
    SPADES("(S)");


    private final String sign;

    Color(String sign) {
        this.sign = sign;
    }

    @Override
    public String toString() {
        return sign;
    }
}
