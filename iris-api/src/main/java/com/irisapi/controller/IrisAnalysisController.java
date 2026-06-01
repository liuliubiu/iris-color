package com.irisapi.controller;

import com.irisapi.service.VisionClientService;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class IrisAnalysisController {

    private final VisionClientService visionClientService;

    public IrisAnalysisController(VisionClientService visionClientService) {
        this.visionClientService = visionClientService;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "ok");
        result.put("service", "iris-api");
        try {
            result.put("vision", visionClientService.health());
        } catch (RestClientException ex) {
            result.put("vision", Map.of("status", "unavailable", "error", ex.getMessage()));
        }
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/iris/analyze", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> analyze(@RequestParam("file") MultipartFile file) {
        return analyzeInternal(file, null);
    }

    @PostMapping(value = "/iris/analyze/manual", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> analyzeManual(
            @RequestParam("file") MultipartFile file,
            @RequestParam("manual_params") String manualParams) {
        return analyzeInternal(file, manualParams);
    }

    private ResponseEntity<String> analyzeInternal(MultipartFile file, String manualParams) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body("{\"success\":false,\"error\":\"empty_file\"}");
        }

        try {
            String response = manualParams == null
                    ? visionClientService.analyze(file)
                    : visionClientService.analyzeManual(file, manualParams);
            return ResponseEntity.ok(response);
        } catch (HttpClientErrorException | HttpServerErrorException ex) {
            return ResponseEntity.status(ex.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(ex.getResponseBodyAsString());
        } catch (RestClientResponseException ex) {
            return ResponseEntity.status(ex.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(ex.getResponseBodyAsString());
        } catch (IOException ex) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("{\"success\":false,\"error\":\"read_upload_failed\"}");
        } catch (RestClientException ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("{\"success\":false,\"error\":\"vision_service_unavailable\"}");
        }
    }
}
