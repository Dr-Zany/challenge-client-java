package com.zuehlke.jasschallenge.game.cards;

public enum Color {
    DIAMONDS("(D)"),
    HEARTS("(H)"),
    SPADES("(S)"),
    CLUBS("(C)");


    private final String sign;

    Color(String sign) {
        this.sign = sign;
    }

    @Override
    public String toString() {
        return sign;
    }
}
