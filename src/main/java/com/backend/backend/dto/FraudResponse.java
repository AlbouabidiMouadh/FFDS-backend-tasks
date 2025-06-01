package com.backend.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class FraudResponse {
    @JsonProperty("isFraud")
    private boolean fraud;

    private double probability;

    public FraudResponse() {
    }

    public FraudResponse(boolean fraud, double probability) {
        this.fraud = fraud;
        this.probability = probability;
    }

    public boolean isFraud() {
        return fraud;
    }

    public void setFraud(boolean fraud) {
        this.fraud = fraud;
    }

    public double getProbability() {
        return probability;
    }

    public void setProbability(double probability) {
        this.probability = probability;
    }
}