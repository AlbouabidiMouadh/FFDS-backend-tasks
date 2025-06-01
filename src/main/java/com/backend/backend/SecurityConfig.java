 // filepath: c:\Users\Lenovo\financialfrauddetectionsys-backend\backend\src\main\java\com\backend\backend\config\SecurityConfig.java
package com.backend.backend;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http.csrf(csrf -> csrf.disable())
            .authorizeHttpRequests(authorize -> authorize.anyRequest().permitAll());
        return http.build();
    }
}