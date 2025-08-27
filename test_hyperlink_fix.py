#!/usr/bin/env python3
"""
Test script to verify the hyperlink parsing fix for SemanticDocumentParser.

This script demonstrates that the enhanced metadata_parser correctly converts
plain text links to markdown format, fixing the issue where AI assistants
would default to adding 'www.example.com' domains.

Usage:
    python test_hyperlink_fix.py

Expected Output:
    - Shows before/after comparison
    - Demonstrates proper markdown link formatting
    - Confirms fix is working correctly
"""

import re
import sys
import os

# Add the SemanticDocumentParser to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SemanticDocumentParser'))

def test_link_detection():
    """Test the hyperlink parsing fix with mock data"""
    
    print("🔧 HYPERLINK PARSING FIX TEST")
    print("=" * 50)
    
    # Import the functions we need to test
    try:
        from element_parsers.metadata_parser import metadata_parser, _manual_link_detection
        print("✅ Successfully imported metadata_parser functions")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Mock element class for testing
    class MockMetadata:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockElement:
        def __init__(self, text, metadata_dict):
            self.text = text
            self.metadata = MockMetadata(**metadata_dict)
    
    # Test case 1: Element with link_texts and link_urls (the problematic case)
    print("\n📊 TEST CASE: Element with link_texts/link_urls metadata")
    print("-" * 50)
    
    element = MockElement(
        text="Content can be found at this link: Excel file with data and graphs",
        metadata_dict={
            "link_texts": ["Excel file with data and graphs"],
            "link_urls": ["https://u-york-eclass.catalyst-ca.net/mod/forum/view.php?id=3497166"],
            "links": None  # This is why the original parser failed
        }
    )
    
    print(f"BEFORE: {element.text}")
    
    # Apply the enhanced metadata parser
    metadata_parser([element])
    
    print(f"AFTER:  {element.text}")
    
    # Check if the fix worked
    expected_link = "[Excel file with data and graphs](https://u-york-eclass.catalyst-ca.net/mod/forum/view.php?id=3497166)"
    if expected_link in element.text:
        print("✅ SUCCESS: Link correctly converted to markdown format!")
        print("🎯 AI assistants will now see the real URL instead of adding 'www.example.com'")
        return True
    else:
        print("❌ FAILURE: Link was not properly converted")
        print(f"Expected to find: {expected_link}")
        return False

def test_backward_compatibility():
    """Test that the fix doesn't break existing functionality"""
    
    print("\n🔄 BACKWARD COMPATIBILITY TEST")
    print("-" * 50)
    
    try:
        from element_parsers.metadata_parser import metadata_parser
        
        # Mock element class
        class MockMetadata:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class MockElement:
            def __init__(self, text, metadata_dict):
                self.text = text
                self.metadata = MockMetadata(**metadata_dict)
        
        # Test case: Element without any links
        element = MockElement(
            text="This is just regular text without any links",
            metadata_dict={}
        )
        
        original_text = element.text
        metadata_parser([element])
        
        if element.text == original_text:
            print("✅ SUCCESS: Non-link elements remain unchanged")
            return True
        else:
            print("❌ FAILURE: Non-link elements were modified unexpectedly")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during backward compatibility test: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧪 TESTING HYPERLINK PARSING FIX")
    print("🎯 Goal: Fix AI assistant 'www.example.com' link issue")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_link_detection()
    test2_passed = test_backward_compatibility()
    
    print("\n📋 FINAL RESULTS")
    print("-" * 30)
    print(f"✅ Link Detection Fix: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Backward Compatibility: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 The fix is ready for deployment")
        print("\n📝 What this fix does:")
        print("   • Detects links in link_texts/link_urls metadata")
        print("   • Converts plain text to markdown format: [text](URL)")
        print("   • Prevents AI assistants from adding 'www.example.com'")
        print("   • Maintains backward compatibility")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please review the implementation before deployment")

if __name__ == "__main__":
    main()
