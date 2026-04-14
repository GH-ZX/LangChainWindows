# Test script to check Ollama and database connectivity
import subprocess
import sys

def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama not available")
            return False
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

def check_model():
    try:
        result = subprocess.run(["ollama", "show", "llama3"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Model llama3 is available")
            return True
        else:
            print("❌ Model llama3 not found")
            return False
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def check_database():
    try:
        from sqlalchemy import create_engine
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/company_db")
        connection = engine.connect()
        connection.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking system requirements...\n")

    ollama_ok = check_ollama()
    model_ok = check_model() if ollama_ok else False
    db_ok = check_database()

    print(f"\n📊 Summary:")
    print(f"Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"Model: {'✅' if model_ok else '❌'}")
    print(f"Database: {'✅' if db_ok else '❌'}")

    if ollama_ok and model_ok and db_ok:
        print("\n🎉 All systems ready! You can run the app.")
    else:
        print("\n⚠️  Some issues found. Please fix them before running the app.")
        print("\nTo fix:")
        if not ollama_ok:
            print("- Install Ollama from https://ollama.ai")
            print("- Run: ollama serve")
        if not model_ok:
            print("- Run: ollama pull llama3")
        if not db_ok:
            print("- Make sure PostgreSQL is running")
            print("- Create database 'company_db'")
            print("- Check connection settings")