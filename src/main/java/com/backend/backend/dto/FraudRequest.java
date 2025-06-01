 package com.backend.backend.dto;
import lombok.Data;

@Data
public class FraudRequest {
    private double amount;
    private int day;
    private String type;
    private String transaction_pair_code;
    private String part_of_the_day;
}