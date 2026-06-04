package com.irisapi.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.util.UriComponentsBuilder;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@Service
public class VisionClientService {

    private final RestTemplate restTemplate;
    private final String baseUrl;

    public VisionClientService(RestTemplate restTemplate, @Value("${iris.vision.base-url}") String baseUrl) {
        this.restTemplate = restTemplate;
        this.baseUrl = baseUrl;
    }

    /**
     * 将上传图片转发至 iris-vision /analyze，原样返回 JSON 字符串。
     */
    public String analyze(MultipartFile file, boolean skipQuality, String detectionMode) throws IOException {
        return postAnalyze(file, "/analyze", null, skipQuality, detectionMode);
    }

    /**
     * 将上传图片与人工调整参数转发至 iris-vision /analyze/manual，原样返回 JSON 字符串。
     */
    public String analyzeManual(
            MultipartFile file,
            String manualParams,
            boolean skipQuality,
            String detectionMode) throws IOException {
        return postAnalyze(file, "/analyze/manual", manualParams, skipQuality, detectionMode);
    }

    private String postAnalyze(
            MultipartFile file,
            String path,
            String manualParams,
            boolean skipQuality,
            String detectionMode) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                String name = file.getOriginalFilename();
                return name != null ? name : "upload.jpg";
            }
        });
        if (manualParams != null) {
            body.add("manual_params", manualParams);
        }

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        String url = UriComponentsBuilder.fromUriString(baseUrl + path)
                .queryParam("skip_quality", skipQuality)
                .queryParam("detection_mode", detectionMode)
                .toUriString();
        return restTemplate.postForObject(url, request, String.class);
    }

    /**
     * 检查 iris-vision 健康状态。
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> health() {
        return restTemplate.getForObject(baseUrl + "/health", Map.class);
    }
}
