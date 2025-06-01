package com.backend.backend.util;

import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;

public class PythonInvoker {
    public static String runPythonScript(String inputJson) throws IOException, InterruptedException {
        File inputFile = File.createTempFile("fraud_input", ".json");
        File outputFile = File.createTempFile("fraud_output", ".json");

        Files.writeString(inputFile.toPath(), inputJson, StandardCharsets.UTF_8);

        ProcessBuilder pb = new ProcessBuilder(
            "python", "predict.py", inputFile.getAbsolutePath(), outputFile.getAbsolutePath()
        );
        pb.directory(new File("src/main/resources")); // Your folder for predict.py
        pb.redirectErrorStream(true);

        Process process = pb.start();
        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Python script failed with exit code: " + exitCode);
        }

        return Files.readString(outputFile.toPath());
    }
}