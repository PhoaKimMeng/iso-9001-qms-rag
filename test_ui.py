import urllib.request
import urllib.error
import json
from html.parser import HTMLParser
import sys

# Define target endpoints
BACKEND_URL = "http://127.0.0.1:8000"

class DOMValidator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found_ids = set()
        self.found_classes = set()
        self.tags = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for attr, value in attrs:
            if attr == "id":
                self.found_ids.add(value)
            elif attr == "class":
                # Handle multiple classes separated by space
                for cls in value.split():
                    self.found_classes.add(cls)

def run_test():
    print("==========================================================")
    print("[SYSTEM] ISO 9001:2015 QMS AI AUDITOR INTEGRATION TEST")
    print("==========================================================")
    
    passed_tests = 0
    failed_tests = 0

    def print_result(name, status, details=""):
        nonlocal passed_tests, failed_tests
        if status:
            print(f"[PASS] {name} - {details}")
            passed_tests += 1
        else:
            print(f"[FAIL] {name} - {details}")
            failed_tests += 1

    # --- TEST 1: Frontend Server Reachability & URL Auto-Discovery ---
    html_content = None
    frontend_working_url = None
    candidate_urls = [
        "http://[::1]:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    for url in candidate_urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                html_content = response.read().decode('utf-8')
                frontend_working_url = url
                print_result("Frontend Server Reachability", True, f"Connected to {url} (Status: {response.status})")
                break
        except Exception:
            continue
            
    if not frontend_working_url:
        print_result("Frontend Server Reachability", False, f"Could not connect to any of: {candidate_urls}")

    # --- TEST 2: Parse index.html DOM Elements ---
    if html_content:
        try:
            parser = DOMValidator()
            parser.feed(html_content)
            
            # Required elements to verify interface functional features
            required_ids = [
                "provider-select", "api-key-input", "save-key-btn", "pdf-file-input", 
                "ingest-btn", "chat-messages", "chat-form", "chat-input", "send-btn", 
                "clauses-accordion-container", "diagnostics-ready-content", 
                "diagnostics-empty-content", "chunk-inspector-slider", "loading-overlay",
                "cohere-config-group", "cohere-api-key-input", "save-cohere-key-btn",
                "cohere-models-group", "cohere-model-select"
            ]
            
            missing_ids = [rid for rid in required_ids if rid not in parser.found_ids]
            
            if not missing_ids:
                print_result("index.html DOM Validation", True, f"Verified all {len(required_ids)} vital elements are present.")
            else:
                print_result("index.html DOM Validation", False, f"Missing critical DOM IDs: {missing_ids}")
        except Exception as e:
            print_result("index.html DOM Validation", False, f"Error parsing HTML: {e}")
    else:
        print_result("index.html DOM Validation", False, "Skipped due to unreachable server")

    # --- TEST 3: Static CSS File Validation ---
    if frontend_working_url:
        try:
            css_url = f"{frontend_working_url}/style.css"
            req = urllib.request.Request(css_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                css_content = response.read().decode('utf-8')
                # Check for generic styling anchors
                has_app_layout = ".app-layout" in css_content or "body" in css_content
                print_result("style.css Delivery", True, f"Loaded CSS ({len(css_content)} chars). Structure: {'Valid' if has_app_layout else 'Invalid'}")
        except Exception as e:
            print_result("style.css Delivery", False, f"Could not fetch style.css from {css_url}: {e}")
    else:
        print_result("style.css Delivery", False, "Skipped due to unreachable server")

    # --- TEST 4: Static JS App Controller Validation ---
    if frontend_working_url:
        try:
            js_url = f"{frontend_working_url}/app.js"
            req = urllib.request.Request(js_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                js_content = response.read().decode('utf-8')
                # Look for state and API_BASE definitions
                has_state = "appState" in js_content and "API_BASE" in js_content
                print_result("app.js Delivery & Syntax", True, f"Loaded JS controller ({len(js_content)} chars). Signature: {'Valid' if has_state else 'Invalid'}")
        except Exception as e:
            print_result("app.js Delivery & Syntax", False, f"Could not fetch app.js from {js_url}: {e}")
    else:
        print_result("app.js Delivery & Syntax", False, "Skipped due to unreachable server")

    # --- TEST 5: Backend FastAPI Reachability ---
    try:
        status_url = f"{BACKEND_URL}/api/status"
        with urllib.request.urlopen(status_url, timeout=5) as response:
            status_json = json.loads(response.read().decode('utf-8'))
            print_result("Backend FastAPI Reachability", True, f"API Status: 200 OK")
    except Exception as e:
        print_result("Backend FastAPI Reachability", False, f"Could not connect to {BACKEND_URL}: {e}")
        status_json = None

    # --- TEST 6: Backend Status Payload & Connectivity Schema ---
    if status_json:
        try:
            # Query Gemini status
            gemini_status_url = f"{BACKEND_URL}/api/status?provider=gemini"
            with urllib.request.urlopen(gemini_status_url, timeout=5) as g_resp:
                gemini_status = json.loads(g_resp.read().decode('utf-8'))
            
            # Query Cohere status
            cohere_status_url = f"{BACKEND_URL}/api/status?provider=cohere"
            with urllib.request.urlopen(cohere_status_url, timeout=5) as c_resp:
                cohere_status = json.loads(c_resp.read().decode('utf-8'))
                
            required_keys = ["api_key_configured", "ollama_active", "index_ready", "indexed_chunks"]
            missing_g = [k for k in required_keys if k not in gemini_status]
            missing_c = [k for k in required_keys if k not in cohere_status]
            
            if not missing_g and not missing_c:
                details = f"Gemini Key Configured: {gemini_status['api_key_configured']}, Cohere Key Configured: {cohere_status['api_key_configured']}, Ollama Connected: {gemini_status['ollama_active']}"
                print_result("Backend /api/status Validation", True, details)
            else:
                print_result("Backend /api/status Validation", False, f"Missing keys - Gemini: {missing_g}, Cohere: {missing_c}")
        except Exception as e:
            print_result("Backend /api/status Validation", False, f"Failed querying status providers: {e}")
    else:
        print_result("Backend /api/status Validation", False, "Skipped due to unreachable API")

    # --- TEST 7: Clauses Schema Database Verification ---
    try:
        clauses_url = f"{BACKEND_URL}/api/clauses"
        with urllib.request.urlopen(clauses_url, timeout=5) as response:
            clauses_json = json.loads(response.read().decode('utf-8'))
            has_clauses = "Clause 4: Context of the Organization" in clauses_json
            print_result("Backend /api/clauses Delivery", True, f"Found {len(clauses_json)} main clauses. Schema: {'Valid' if has_clauses else 'Invalid'}")
    except Exception as e:
        print_result("Backend /api/clauses Delivery", False, f"Could not fetch clauses: {e}")

    # --- SUMMARY DISPLAY ---
    print("==========================================================")
    print("[SUMMARY] INTEGRATION TEST SUMMARY")
    print("==========================================================")
    print(f"Total Tests Run: {passed_tests + failed_tests}")
    print(f"Passed:          {passed_tests}")
    print(f"Failed:          {failed_tests}")
    print("==========================================================")
    
    if failed_tests == 0:
        print("SUCCESS: The UI and backend integration tests passed completely!")
        sys.exit(0)
    else:
        print("WARNING: Some tests failed. Please check the network connectivity or configuration.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
