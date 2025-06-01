package com.backend.backend.controller;

import com.backend.backend.dto.FraudRequest;
import com.backend.backend.dto.FraudResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/api")
public class FraudDetectionController {

    private static final List<String> VALID_TYPES = Arrays.asList("CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN",
            "DEBIT");
    private static final List<String> VALID_PAIR_CODES = Arrays.asList("cc", "cm");
    private static final List<String> VALID_PARTS_OF_DAY = Arrays.asList("morning", "afternoon", "evening", "night");
    private static final String WORKING_DIR = "D:\\GitHub\\local\\FFDS-backend\\src\\main\\resources";

    @PostMapping("/fraud-detect")
    public ResponseEntity<?> detectFraud(@RequestBody FraudRequest request) {
        // Validate input fields
        if (request == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Request body is null");
        }
        if (!VALID_TYPES.contains(request.getType())) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Invalid transaction type: " + request.getType());
        }
        if (!VALID_PAIR_CODES.contains(request.getTransaction_pair_code())) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Invalid transaction_pair_code: " + request.getTransaction_pair_code());
        }
        if (!VALID_PARTS_OF_DAY.contains(request.getPart_of_the_day())) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Invalid part_of_the_day: " + request.getPart_of_the_day());
        }
        if (request.getAmount() <= 0) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Amount must be positive: " + request.getAmount());
        }
        if (request.getDay() < 1 || request.getDay() > 31) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Day must be between 1 and 31: " + request.getDay());
        }

        try {
            ObjectMapper mapper = new ObjectMapper();
            String inputJson = mapper.writeValueAsString(request);

            // Build the command (platform-agnostic)
            String pythonCommand = System.getProperty("os.name").toLowerCase().contains("win") ? "python" : "python3";
            ProcessBuilder pb = new ProcessBuilder(pythonCommand, "predict.py");
            pb.directory(new File(WORKING_DIR));
            pb.redirectErrorStream(false); // Keep stderr separate
            Process process = pb.start();

            // Write JSON to stdin
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
                writer.write(inputJson);
                writer.flush();
            }

            // Read stdout
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line);
                }
            }

            // Read stderr
            StringBuilder errorOutput = new StringBuilder();
            try (BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                String line;
                while ((line = errorReader.readLine()) != null) {
                    errorOutput.append(line).append("\n");
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body("Python script exited with code: " + exitCode + ". Error: " + errorOutput.toString());
            }

            // Parse output to FraudResponse
            if (output.length() == 0) {
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body("Python script returned no output");
            }
            FraudResponse fraudResponse = mapper.readValue(output.toString(), FraudResponse.class);
            return ResponseEntity.ok(fraudResponse);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error during fraud detection: " + e.getMessage());
        }
    }
}