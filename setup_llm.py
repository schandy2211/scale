#!/usr/bin/env python3
"""
Simple setup script for LLM integration.
"""

import os
import sys

def setup_openai_key():
    """Setup OpenAI API key."""
    print("Setting up OpenAI API key for LLM integration...")
    
    # Check if already set
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY is already set in environment")
        return True
    
    # Try to get from user input
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️  No API key provided. You can set it later with:")
        print("   export OPENAI_API_KEY=your_key_here")
        return False
    
    # Set for current session
    os.environ["OPENAI_API_KEY"] = api_key
    print("✓ API key set for current session")
    
    # Ask if they want to add to shell profile
    add_to_profile = input("Add to shell profile for permanent setup? (y/n): ").strip().lower()
    
    if add_to_profile in ['y', 'yes']:
        shell_profile = os.path.expanduser("~/.zshrc") if "zsh" in os.environ.get("SHELL", "") else os.path.expanduser("~/.bashrc")
        
        try:
            with open(shell_profile, "a") as f:
                f.write(f"\n# OpenAI API Key for SCALE\n")
                f.write(f"export OPENAI_API_KEY={api_key}\n")
            print(f"✓ Added to {shell_profile}")
            print("Run 'source ~/.zshrc' (or ~/.bashrc) to reload your shell")
        except Exception as e:
            print(f"⚠️  Could not write to {shell_profile}: {e}")
            print("You can manually add this line to your shell profile:")
            print(f"export OPENAI_API_KEY={api_key}")
    
    return True

def test_llm_integration():
    """Test if LLM integration works."""
    print("\nTesting LLM integration...")
    
    try:
        from agent.llm_controller import LLMController
        from agent.llm_agent import LLMAgent
        
        # Test controller
        controller = LLMController()
        print("✓ LLM Controller initialized successfully")
        
        # Test agent
        agent = LLMAgent()
        print("✓ LLM Agent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM integration test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("SCALE LLM Integration Setup")
    print("=" * 30)
    
    # Setup API key
    api_key_set = setup_openai_key()
    
    if api_key_set:
        # Test integration
        if test_llm_integration():
            print("\n🎉 LLM integration setup complete!")
            print("\nYou can now run SCALE with LLM controller:")
            print("  python baseline/demo.py --llm")
            print("  python baseline/baseline_opt.py --llm --agent")
        else:
            print("\n⚠️  Setup completed but LLM integration test failed")
            print("Check your API key and internet connection")
    else:
        print("\n⚠️  Setup incomplete - no API key provided")
        print("Run this script again when you have an API key")

if __name__ == "__main__":
    main()
