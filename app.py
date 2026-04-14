import sys
import json
import os
import subprocess
import time

# --- مكتبات الواجهة الرسومية ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTextEdit, QStackedWidget, QMessageBox, QFrame, 
                               QComboBox, QGraphicsOpacityEffect, QSizePolicy, QFormLayout, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QSize, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor, QFontDatabase

# --- مكتبات الذكاء الاصطناعي ---
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# =================================================================================
# 1. كلاس تشغيل Ollama في الخلفية (Ollama Auto-Runner)
# =================================================================================
class OllamaLoaderThread(QThread):
    finished_signal = Signal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            # تشغيل أمر ollama run في الخلفية دون فتح نافذة
            # ملاحظة: هذا الأمر يتأكد من أن الموديل محمل في الذاكرة
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # نرسل طلب "pull" أولاً للتأكد من وجوده، أو run مباشرة
            # هنا نستخدم Popen لتشغيله وتركه يعمل
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # ننتظر قليلاً ثم نغلق العملية لأننا فقط نريد إيقاظ الموديل
            # (LangChain سيتصل عبر API، لسنا بحاجة لبقاء الـ process مفتوحاً للأبد في التيرمينال)
            time.sleep(2) 
            # نرسل إشارة نجاح
            self.finished_signal.emit(f"تم تحميل الموديل {self.model_name} بنجاح.")
            
        except Exception as e:
            self.finished_signal.emit(f"تنبيه: تأكد من تشغيل Ollama. ({str(e)})")

# =================================================================================
# 2. كلاس معالجة الشات (Chat Worker)
# =================================================================================
class ChatWorker(QThread):
    response_received = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, settings, question):
        super().__init__()
        self.settings = settings
        self.question = question

    def get_schema(self, db):
        return db.get_table_info()

    def clean_sql_query(self, text):
        text = text.replace("```sql", "").replace("```", "").strip()
        if "SQL Query:" in text:
            text = text.split("SQL Query:")[1]
        return text.strip()

    def run(self):
        try:
            print(f"Starting chat worker with question: {self.question}")
            
            # Check if Ollama is available
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    raise Exception(f"Ollama not available: {result.stderr}")
                print("Ollama is available")
            except Exception as ollama_err:
                print(f"Ollama check failed: {str(ollama_err)}")
                self.error_occurred.emit(f"خطأ في Ollama: {str(ollama_err)}")
                return
            
            # Check if the model is available
            try:
                model_check = subprocess.run(["ollama", "show", self.settings['model']], 
                                           capture_output=True, text=True, timeout=10)
                if model_check.returncode != 0:
                    raise Exception(f"Model {self.settings['model']} not found")
                print(f"Model {self.settings['model']} is available")
            except Exception as model_err:
                print(f"Model check failed: {str(model_err)}")
                self.error_occurred.emit(f"النموذج {self.settings['model']} غير متوفر: {str(model_err)}")
                return
            
            db_uri = f"postgresql+psycopg2://{self.settings['user']}:{self.settings['pass']}@{self.settings['host']}:{self.settings['port']}/{self.settings['dbname']}"
            print(f"Database URI: {db_uri}")
            
            try:
                db = SQLDatabase.from_uri(db_uri)
                print("Database connection successful")
            except Exception as db_err:
                print(f"Database connection failed: {str(db_err)}")
                self.error_occurred.emit(f"فشل الاتصال بقاعدة البيانات: {str(db_err)}")
                return
            
            try:
                llm = ChatOllama(model=self.settings['model'], temperature=0)
                print(f"LLM initialized with model: {self.settings['model']}")
            except Exception as llm_err:
                print(f"LLM initialization failed: {str(llm_err)}")
                self.error_occurred.emit(f"فشل تهيئة النموذج اللغوي: {str(llm_err)}")
                return

            sql_prompt = ChatPromptTemplate.from_template(
                """Based on the table schema below, write a syntacticall correct PostgreSQL query that would answer the user's question.
                return ONLY the SQL query, nothing else. Do not use Markdown.
                Schema: {schema}
                Question: {question}
                SQL Query:"""
            )

            sql_gen_chain = (
                RunnablePassthrough.assign(schema=lambda _: self.get_schema(db))
                | sql_prompt | llm | StrOutputParser()
            )

            print("Generating SQL query...")
            try:
                raw_query = sql_gen_chain.invoke({"question": self.question})
                print(f"Raw query: {raw_query}")
            except Exception as query_err:
                print(f"SQL query generation failed: {str(query_err)}")
                self.error_occurred.emit(f"فشل في توليد الاستعلام SQL: {str(query_err)}")
                return
            
            clean_query = self.clean_sql_query(raw_query)
            print(f"Clean query: {clean_query}")

            try:
                print("Executing SQL query...")
                sql_result = db.run(clean_query)
                print(f"SQL result: {sql_result}")
            except Exception as db_err:
                print(f"Database error: {str(db_err)}")
                sql_result = f"Error executing SQL: {str(db_err)}"

            final_prompt = ChatPromptTemplate.from_template(
                """Given the following user question, corresponding SQL query, and SQL result, answer the user question in Arabic.
                Be professional, concise, and formal.

                Question: {question}
                SQL Query: {query}
                SQL Result: {result}
                Answer (in Arabic):"""
            )

            print("Generating final response...")
            try:
                final_chain = (final_prompt | llm | StrOutputParser())
                final_response = final_chain.invoke({
                    "question": self.question, "query": clean_query, "result": sql_result
                })
                print(f"Final response: {final_response}")
            except Exception as response_err:
                print(f"Final response generation failed: {str(response_err)}")
                self.error_occurred.emit(f"فشل في توليد الرد النهائي: {str(response_err)}")
                return
            
            self.response_received.emit(final_response)

        except Exception as e:
            print(f"Chat worker error: {str(e)}")
            self.error_occurred.emit(f"خطأ غير متوقع: {str(e)}")

# =================================================================================
# 3. كلاس توليد الاقتراحات (Suggestion Worker)
# =================================================================================
class SuggestionWorker(QThread):
    suggestions_ready = Signal(list)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def get_schema(self, db):
        return db.get_table_info()

    def run(self):
        try:
            db_uri = f"postgresql+psycopg2://{self.settings['user']}:{self.settings['pass']}@{self.settings['host']}:{self.settings['port']}/{self.settings['dbname']}"
            db = SQLDatabase.from_uri(db_uri)
            llm = ChatOllama(model=self.settings['model'], temperature=0.7)

            suggestion_prompt = ChatPromptTemplate.from_template(
                """Based on the table schema below, generate 4 diverse, insightful, and short example questions in Arabic that a user might ask.
                Return ONLY a JSON list of strings. Do not use Markdown.
                Example: ["ما هو متوسط الرواتب؟", "كم عدد الموظفين في كل قسم؟"]

                Schema: {schema}
                
                JSON List of Questions:"""
            )

            suggestion_chain = (
                RunnablePassthrough.assign(schema=lambda _: self.get_schema(db))
                | suggestion_prompt | llm | StrOutputParser()
            )

            response_text = suggestion_chain.invoke({})
            # Clean and parse the JSON response
            clean_response = response_text.strip().replace("```json", "").replace("```", "")
            suggestions = json.loads(clean_response)
            self.suggestions_ready.emit(suggestions)

        except Exception as e:
            # Don't emit errors to the user for this, just fail silently
            print(f"Suggestion generation failed: {e}")
            self.suggestions_ready.emit([]) 

# =================================================================================
# 6. كلاس معالجة المحادثة العامة (General Chat Worker)
# =================================================================================
class GeneralChatWorker(QThread):
    response_received = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, settings, question, conversation_history=None):
        super().__init__()
        self.settings = settings
        self.question = question
        self.conversation_history = conversation_history or []

    def run(self):
        try:
            print(f"Starting general chat worker with question: {self.question}")
            
            # Check if Ollama is available
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    raise Exception(f"Ollama not available: {result.stderr}")
                print("Ollama is available")
            except Exception as ollama_err:
                print(f"Ollama check failed: {str(ollama_err)}")
                self.error_occurred.emit(f"خطأ في Ollama: {str(ollama_err)}")
                return
            
            # Check if the model is available
            try:
                model_check = subprocess.run(["ollama", "show", self.settings['model']], 
                                           capture_output=True, text=True, timeout=10)
                if model_check.returncode != 0:
                    raise Exception(f"Model {self.settings['model']} not found")
                print(f"Model {self.settings['model']} is available")
            except Exception as model_err:
                print(f"Model check failed: {str(model_err)}")
                self.error_occurred.emit(f"النموذج {self.settings['model']} غير متوفر: {str(model_err)}")
                return
            
            try:
                llm = ChatOllama(model=self.settings['model'], temperature=0.7)
                print(f"LLM initialized with model: {self.settings['model']}")
            except Exception as llm_err:
                print(f"LLM initialization failed: {str(llm_err)}")
                self.error_occurred.emit(f"فشل تهيئة النموذج اللغوي: {str(llm_err)}")
                return

            # Build conversation history
            messages = []
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages
                if msg['role'] == 'user':
                    messages.append(("human", msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(("assistant", msg['content']))
            
            messages.append(("human", self.question))

            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "أنت مساعد ذكي مفيد. أجب باللغة العربية بشكل ودود ومفيد. كن موجزاً ولكن شاملاً في إجاباتك."),
                *messages
            ])

            print("Generating general response...")
            try:
                final_chain = general_prompt | llm | StrOutputParser()
                final_response = final_chain.invoke({})
                print(f"General response: {final_response}")
            except Exception as response_err:
                print(f"General response generation failed: {str(response_err)}")
                self.error_occurred.emit(f"فشل في توليد الرد: {str(response_err)}")
                return
            
            self.response_received.emit(final_response)

        except Exception as e:
            print(f"General chat worker error: {str(e)}")
            self.error_occurred.emit(f"خطأ غير متوقع: {str(e)}")
class SuggestionButton(QPushButton):
    def __init__(self, text, accent_color, bg_color):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: 1px solid {accent_color};
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 13px;
                margin: 0 5px;
            }}
            QPushButton:hover {{
                background-color: {accent_color};
                color: #002b28;
            }}
        """)
# =================================================================================
# 4. الإعدادات الافتراضية (Default Configuration)
# =================================================================================
DEFAULT_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "company_db",
    "user": "postgres",
    "pass": "admin",
    "model": "llama3"
}

# =================================================================================
# 4. إدارة محفوظات المحادثات (Chat History Manager)
# =================================================================================
class ChatHistoryManager:
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.ensure_history_file_exists()
    
    def ensure_history_file_exists(self):
        """تأكد من وجود ملف المحفوظات، أنشئ ملفاً جديداً إن لم يكن موجوداً"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({"conversations": []}, f, ensure_ascii=False, indent=2)
    
    def save_conversation(self, conversation_data):
        """حفظ محادثة جديدة في السجل"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # إضافة المحادثة الجديدة
            conversation_data['id'] = len(data['conversations']) + 1
            conversation_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            data['conversations'].append(conversation_data)
            
            # حفظ البيانات المحدثة
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"تم حفظ المحادثة: {conversation_data.get('title', 'بدون عنوان')}")
            return True
        except Exception as e:
            print(f"خطأ في حفظ المحادثة: {e}")
            return False
    
    def load_all_conversations(self):
        """تحميل جميع المحادثات السابقة"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('conversations', [])
        except Exception as e:
            print(f"خطأ في تحميل المحادثات: {e}")
            return []
    
    def get_conversation_by_id(self, conv_id):
        """الحصول على محادثة معينة بواسطة المعرف"""
        conversations = self.load_all_conversations()
        for conv in conversations:
            if conv.get('id') == conv_id:
                return conv
        return None
    
    def delete_conversation(self, conv_id):
        """حذف محادثة من السجل"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['conversations'] = [c for c in data['conversations'] if c.get('id') != conv_id]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"تم حذف المحادثة رقم {conv_id}")
            return True
        except Exception as e:
            print(f"خطأ في حذف المحادثة: {e}")
            return False

# =================================================================================
# 5. الواجهة الرئيسية (Main App)
# =================================================================================
class ModernApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("نظام تحليل الموظفين")
        self.resize(900, 650)  # حجم أصغر ومتوسط
        self.setMinimumSize(700, 500)  # حد أدنى للحجم لضمان قابلية الاستخدام
        self.settings_file = "config.json"
        
        # Initialize config file if it doesn't exist
        self.ensure_config_exists()
        
        # Set layout direction for Arabic support
        self.setLayoutDirection(Qt.RightToLeft)
        
        # الألوان (الهوية البصرية المحسنة)
        self.colors = {
            "bg_dark": "#002b28",       # خلفية الشريط الجانبي
            "bg_main": "#003B36",       # الخلفية الرئيسية
            "bg_card": "#004D4A",       # خلفية البطاقات/الحقول
            "accent": "#C5A059",        # ذهبي
            "accent_hover": "#D4B06A",  # ذهبي فاتح
            "text": "#C3A674",
            "text_dim": "#AAAAAA"
        }
        
        # مؤشر التحميل وفقاعات المحادثة
        self.loading_indicator_html = "<div align='center' style='color: #888; font-style: italic;'>جاري معالجة الاستعلام...</div>"

        # متغيرات للمحادثة العامة
        self.general_chat_history = []
        
        # Initialize chat history manager
        self.history_manager = ChatHistoryManager()

        self.init_ui()
        self.load_settings()
        
        # تشغيل الموديل تلقائياً عند الفتح
        self.trigger_ollama_load()

    def add_general_bubble(self, text, bubble_type="user"):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 5, 10, 5)
        row_layout.setSpacing(10)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setLayoutDirection(Qt.RightToLeft)

        # حساب العرض
        viewport_width = self.general_chat_scroll.viewport().width() if self.general_chat_scroll.viewport().width() > 0 else 600
        max_bubble_width = int(viewport_width * 0.80)
        
        font_metrics = bubble.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        bubble_width = min(max(text_width + 40, 100), max_bubble_width)
        bubble.setFixedWidth(bubble_width)

        # الستايل الموحد
        common_style = """
            QLabel {
                font-family: "Segoe UI", "Arial", sans-serif;
                font-size: 16px;
                padding: 10px 15px;
                border-radius: 15px;
                line-height: 1.5;
            }
        """

        if bubble_type == "user":
            # لون المستخدم (ذهبي accent)
            bubble.setStyleSheet(common_style + f"""
                QLabel {{
                    background-color: {self.colors['accent']};
                    color: {self.colors['bg_dark']};  /* نص غامق على خلفية ذهبية */
                    border-bottom-right-radius: 2px;
                }}
            """)
            bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(bubble)
            row_layout.addStretch()

        elif bubble_type == "bot":
            # لون البوت (خلفية الكارد bg_card)
            bubble.setStyleSheet(common_style + f"""
                QLabel {{
                    background-color: {self.colors['bg_card']};
                    color: white; /* نص أبيض */
                    border: 1px solid {self.colors['accent']};
                    border-bottom-left-radius: 2px;
                }}
            """)
            bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addStretch()
            row_layout.addWidget(bubble)

        elif bubble_type == "loading":
             bubble.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.2);
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 12px;
                    font-style: italic;
                    padding: 5px 15px;
                    border-radius: 12px;
                }
             """)
             bubble.setAlignment(Qt.AlignCenter)
             row_layout.addStretch()
             row_layout.addWidget(bubble)
             row_layout.addStretch()

        elif bubble_type == "error":
             bubble.setStyleSheet("""
                QLabel {
                    color: #ffcccc;
                    background-color: rgba(255, 0, 0, 0.2);
                    border: 1px solid #ff4444;
                    font-size: 14px;
                    border-radius: 10px;
                    padding: 8px 15px;
                }
             """)
             bubble.setAlignment(Qt.AlignCenter)
             row_layout.addStretch()
             row_layout.addWidget(bubble)
             row_layout.addStretch()

        self.general_chat_layout.addWidget(row_widget)
        
        QApplication.processEvents()
        QTimer.singleShot(10, lambda: self.general_chat_scroll.verticalScrollBar().setValue(
            self.general_chat_scroll.verticalScrollBar().maximum()))
        
    def toggle_sidebar(self):
        # الحصول على العرض الحالي (نعتمد على maximumWidth للتحكم بالأنميشن)
        current_width = self.sidebar.maximumWidth()
        
        # إذا كان العرض 0 (مخفي)، الهدف 280. وإلا فالهدف 0.
        if current_width == 0:
            new_width = 280
            self.sidebar.setVisible(True) # التأكد من الظهور
        else:
            new_width = 0

        # إعداد الأنميشن
        self.side_anim = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.side_anim.setDuration(300) # المدة بالمللي ثانية
        self.side_anim.setStartValue(current_width)
        self.side_anim.setEndValue(new_width)
        self.side_anim.setEasingCurve(QEasingCurve.InOutCubic)
        
        # تحديث فقاعات المحادثة أثناء الحركة لتبقى متناسقة
        self.side_anim.valueChanged.connect(lambda: self.update_chat_bubbles_size())
        
        # عند الانتهاء، إذا كان العرض 0 نخفي العنصر تماماً للأداء
        self.side_anim.finished.connect(lambda: self.sidebar.setVisible(new_width > 0))
        
        self.side_anim.start()

    def update_chat_bubbles_size(self):
        if not hasattr(self, 'chat_scroll') or not hasattr(self, 'chat_layout'):
            return

        viewport_width = self.chat_scroll.viewport().width()
        if viewport_width <= 0: return # تجنب الأخطاء عند البدء

        max_bubble_width = int(viewport_width * 0.80)

        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                row_widget = item.widget()
                labels = row_widget.findChildren(QLabel)
                for bubble in labels:
                    if bubble.wordWrap(): # هذه هي فقاعات الشات
                        # إعادة حساب العرض
                        font_metrics = bubble.fontMetrics()
                        text_width = font_metrics.horizontalAdvance(bubble.text())
                        
                        # نفس المنطق: العرض حسب النص لكن لا يتجاوز الحد الأقصى
                        new_width = min(max(text_width + 50, 100), max_bubble_width)
                        
                        # إذا وصل للحد الأقصى، نترك الارتفاع يتعدل تلقائياً
                        bubble.setFixedWidth(new_width)
                        bubble.adjustSize()

    def resizeEvent(self, event):
        """
        حدث يتم استدعاؤه تلقائياً عند تغيير حجم النافذة
        نقوم فيه بتحديث عرض الفقاعات لتبقى متناسقة
        """
        super().resizeEvent(event)
        self.update_chat_bubbles_size()
        # التأكد من أن ودجت الشات تم إنشاؤه
        if hasattr(self, 'chat_widget') and hasattr(self, 'chat_layout'):
            # العرض الجديد المتاح للفقاعة (85% من عرض المنطقة)
            new_max_width = int(self.chat_widget.width() * 0.85)
            
            # المرور على جميع العناصر في الشات
            for i in range(self.chat_layout.count()):
                item = self.chat_layout.itemAt(i)
                if item and item.widget():
                    row_widget = item.widget()
                    # البحث عن الليبل داخل صف الفقاعة
                    labels = row_widget.findChildren(QLabel)
                    for label in labels:
                        # نتأكد أننا نعدل فقاعات النصوص فقط وليس أي ليبل آخر
                        # (الفقاعات لديها خاصية wordWrap مفعلة)
                        if label.wordWrap():
                            label.setMaximumWidth(new_max_width)

        # نفس الشيء للمحادثة العامة
        if hasattr(self, 'general_chat_widget') and hasattr(self, 'general_chat_layout'):
            new_max_width_general = int(self.general_chat_widget.width() * 0.85)
            
            for i in range(self.general_chat_layout.count()):
                item = self.general_chat_layout.itemAt(i)
                if item and item.widget():
                    row_widget = item.widget()
                    labels = row_widget.findChildren(QLabel)
                    for label in labels:
                        if label.wordWrap():
                            label.setMaximumWidth(new_max_width_general)

    def init_ui(self):

        main_widget = QWidget()
        
        # 1. تعيين أيقونة للتطبيق والنافذة
        app_icon = QIcon(resource_path("icon.ico"))
        self.setWindowIcon(app_icon)

        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 40, 20, 20)
        sidebar_layout.setSpacing(15)

        # Logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_filename = resource_path("mod_logo.png")
        if os.path.exists(logo_filename):
            pixmap = QPixmap(logo_filename)
            scaled_pixmap = pixmap.scaledToWidth(140, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            logo_label.setText("[شعار الوزارة]")
            logo_label.setObjectName("LogoPlaceholder")
        
        sidebar_layout.addWidget(logo_label)

        # Title
        app_title = QLabel("نظام التحليل الذكي")
        app_title.setObjectName("AppTitle")
        app_title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(app_title)
        
        # Settings Button
        sidebar_layout.addSpacing(20)

        # Navigation Buttons
        self.btn_chat = self.create_nav_button("تحليل البيانات", "💬", 0)
        self.btn_general = self.create_nav_button("مساعد GPT", "🤖", 1)
        self.btn_settings = self.create_nav_button("الإعدادات ", "⚙️", 2)
        self.btn_history = self.create_nav_button("المحفوظات", "📜", 3)

        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_general)
        sidebar_layout.addWidget(self.btn_settings)
        sidebar_layout.addWidget(self.btn_history)
        sidebar_layout.addStretch()

        # Footer
        footer = QLabel("Authorized Personnel Only\nV 2.2.0")
        footer.setObjectName("Footer")
        footer.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(footer)
        
        self.sidebar.setVisible(False) # Start with sidebar hidden

        # --- Content Area ---
        content_area = QFrame()
        content_area.setObjectName("ContentArea")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stacked_widget = QStackedWidget()

        # Create the new two-stage chat page
        main_chat_page = QWidget()
        main_chat_layout = QVBoxLayout(main_chat_page)
        main_chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_stack = QStackedWidget()
        main_chat_layout.addWidget(self.chat_stack)

        self.welcome_page = self.create_welcome_page()
        self.conversation_page = self.create_conversation_page()
        
        self.chat_stack.addWidget(self.welcome_page)
        self.chat_stack.addWidget(self.conversation_page)

        self.page_settings = self.create_settings_page()
        self.page_general = self.create_general_chat_page()
        self.page_history = self.create_history_page()
        
        self.stacked_widget.addWidget(main_chat_page)
        self.stacked_widget.addWidget(self.page_general)
        self.stacked_widget.addWidget(self.page_settings)
        self.stacked_widget.addWidget(self.page_history)

        content_layout.addWidget(self.stacked_widget)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(content_area)

        self.apply_styles()

    def create_nav_button(self, text, icon_char, index):
        btn = QPushButton(f"  {icon_char}  {text}")
        btn.setObjectName("NavButton")
        btn.setCheckable(True)
        # عند النقر نقوم بالتحويل ونضيف انميشن
        btn.clicked.connect(lambda: self.switch_page(index, btn))
        if index == 0: btn.setChecked(True)
        return btn

    def switch_page(self, index, btn_sender):
        # 1. إدارة حالة الأزرار (تجميلي)
        self.btn_chat.setChecked(False)
        self.btn_general.setChecked(False)
        self.btn_settings.setChecked(False)
        self.btn_history.setChecked(False)
        btn_sender.setChecked(True)

        # 2. تحديث صفحة المحفوظات قبل عرضها
        if index == 3:
            self.refresh_history_page()

        # 3. تغيير الصفحة
        self.stacked_widget.setCurrentIndex(index)
        
        # =========================================================
        # الحل الجذري لمشكلة التجميد وتحديث المحتوى
        # =========================================================
        
        # إجبار النظام على معالجة الأحداث المتراكمة فوراً
        QApplication.processEvents()
        
        # الحصول على الصفحة الحالية
        current_page = self.stacked_widget.currentWidget()
        
        # إجبار الـ Layout على إعادة الحساب
        if current_page.layout():
            current_page.layout().invalidate()
            current_page.layout().activate()
            current_page.updateGeometry()

        # إصلاح خاص لصفحة الشات العام (Index 1) لأنها تحتوي على ScrollArea
        if index == 1:
            # إعادة حساب حجم الودجت الداخلي للسكول
            if hasattr(self, 'general_chat_widget'):
                self.general_chat_widget.adjustSize()
                self.general_chat_widget.update()
            
            # تمرير السكرول للأسفل (اختياري، لضمان رؤية آخر رسالة)
            if hasattr(self, 'general_chat_scroll'):
                QTimer.singleShot(50, lambda: self.general_chat_scroll.verticalScrollBar().setValue(
                    self.general_chat_scroll.verticalScrollBar().maximum()))

        # إصلاح خاص لصفحة الشات الأساسي (Index 0)
        if index == 0:
            if hasattr(self, 'chat_widget'):
                self.chat_widget.adjustSize()
                self.chat_widget.update()

        # =========================================================

        # 3. تشغيل انميشن الظهور
        self.fade_in_animation(current_page)
    def fade_in_animation(self, widget):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(400) # 400ms
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()

    def create_welcome_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setAlignment(Qt.AlignCenter)

        layout.addStretch()

        welcome_title = QLabel("نظام التحليل الذكي")
        welcome_title.setObjectName("WelcomeTitle")
        welcome_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_title)
        
        welcome_subtitle = QLabel("ابدأ بطرح سؤالك للاستعلام من قاعدة البيانات")
        welcome_subtitle.setObjectName("WelcomeSubtitle")
        welcome_subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_subtitle)
        layout.addSpacing(30)

        # --- منطقة الإدخال (التعديل هنا) ---
        welcome_input_frame = QFrame()
        welcome_input_frame.setObjectName("WelcomeInputFrame")
        
        # استخدام Layout يضمن تمدد العناصر
        welcome_input_layout = QHBoxLayout(welcome_input_frame)
        welcome_input_layout.setContentsMargins(10, 5, 10, 5) # هوامش داخلية للإطار
        
        self.welcome_input = QLineEdit()
        self.welcome_input.setPlaceholderText("مثال: ما هو عدد الموظفين في كل قسم؟")
        self.welcome_input.setObjectName("WelcomeInput")
        self.welcome_input.returnPressed.connect(self.transition_to_chat)
        self.welcome_input.setLayoutDirection(Qt.RightToLeft)
        
        # !! إصلاح الحجم الصغير !!
        # نجبر الحقل على التمدد أفقياً
        self.welcome_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # نضع حداً أدنى للعرض والارتفاع
        self.welcome_input.setMinimumWidth(400) 
        self.welcome_input.setMinimumHeight(50)

        self.welcome_send_btn = QPushButton("إرسال")
        self.welcome_send_btn.setObjectName("WelcomeSendButton")
        self.welcome_send_btn.setCursor(Qt.PointingHandCursor)
        self.welcome_send_btn.clicked.connect(self.transition_to_chat)
        self.welcome_send_btn.setMinimumHeight(45) # ضمان ارتفاع الزر

        welcome_input_layout.addWidget(self.welcome_input)
        welcome_input_layout.addWidget(self.welcome_send_btn)
        
        # نضع الإطار في المنتصف ولكن نسمح له بالتمدد قليلاً
        layout.addWidget(welcome_input_frame, alignment=Qt.AlignCenter)
        
        # --- نهاية التعديل ---

        # Suggestions Area
        self.suggestions_container = QWidget()
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.suggestions_container)

        layout.addStretch()
        return page

    def transition_to_chat(self):
        question = self.welcome_input.text()
        if not question:
            return

        # 1. تحريك القائمة الجانبية
        self.sidebar.setVisible(True)
        self.sidebar_anim = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.sidebar_anim.setDuration(300)
        self.sidebar_anim.setStartValue(0)
        self.sidebar_anim.setEndValue(280)
        self.sidebar_anim.setEasingCurve(QEasingCurve.InOutCubic)
        # تحديث الفقاعات أثناء حركة القائمة
        self.sidebar_anim.valueChanged.connect(lambda: self.update_chat_bubbles_size())
        self.sidebar_anim.start()

        # 2. نقل النص وتنظيف حقل الترحيب
        self.txt_input.setText(question)
        self.welcome_input.clear()

        # 3. التبديل لصفحة الشات
        self.chat_stack.setCurrentIndex(1)
        
        # 4. تحديث فوري للأبعاد لإصلاح أي تشوه
        QApplication.processEvents() 
        self.update_chat_bubbles_size()

        # 5. تشغيل انميشن ظهور ناعم للصفحة
        self.fade_in_animation(self.conversation_page)
        
        # 6. بدء المعالجة
        self.start_chat_thread()

    # def animate_input_slide(self):
    #     # Get the input frame from conversation page
    #     input_frame = self.conversation_page.findChild(QFrame, "InputFrame")
    #     if input_frame:
    #         # Make sure the input frame is visible
    #         input_frame.setVisible(True)
    #         input_frame.raise_()  # Bring to front
            
    #         # Simple slide animation from slightly above to final position
    #         final_pos = input_frame.pos()
    #         start_pos = final_pos
    #         start_pos.setY(final_pos.y() - 50)  # Start 50px above
            
    #         # Set initial position
    #         input_frame.move(start_pos.x(), start_pos.y())
            
    #         # Create simple slide animation
    #         self.input_slide_anim = QPropertyAnimation(input_frame, b"pos")
    #         self.input_slide_anim.setDuration(400)
    #         self.input_slide_anim.setStartValue(start_pos)
    #         self.input_slide_anim.setEndValue(final_pos)
    #         self.input_slide_anim.setEasingCurve(QEasingCurve.OutCubic)
    #         self.input_slide_anim.start()

    def create_conversation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # 1. الهيدر (العنوان وزر المسح وزر القائمة)
        header_layout = QHBoxLayout()
        
        # --- إضافة زر القائمة الجانبية ---
        self.btn_toggle_menu = QPushButton("≡")
        self.btn_toggle_menu.setObjectName("MenuButton") # سنضيف الستايل لاحقاً
        self.btn_toggle_menu.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_menu.setFixedSize(40, 40)
        self.btn_toggle_menu.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.btn_toggle_menu)
        # -------------------------------

        header = QLabel("محادثة الاستعلام")
        header.setObjectName("HeaderTitle")
        
        self.btn_clear = QPushButton("مسح المحادثة")
        self.btn_clear.setObjectName("ClearButton")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_chat)

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_clear)
        
        layout.addLayout(header_layout)

        # 2. منطقة الشات
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setObjectName("ChatScroll")
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background: transparent;") 
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setSpacing(15)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setAlignment(Qt.AlignTop)
        
        self.chat_scroll.setWidget(self.chat_widget)
        layout.addWidget(self.chat_scroll, stretch=1)

        # 3. منطقة الإدخال
        input_frame = QFrame()
        input_frame.setObjectName("InputFrame")
        input_frame.setFixedHeight(80)
        
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("أدخل استفسارك هنا...")
        self.txt_input.returnPressed.connect(self.start_chat_thread)
        self.txt_input.setLayoutDirection(Qt.RightToLeft)
        self.txt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.btn_send = QPushButton("إرسال الطلب")
        self.btn_send.setObjectName("SendButton")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.clicked.connect(self.on_send_button_clicked)
        self.btn_send.setMinimumHeight(45)

        input_layout.addWidget(self.txt_input)
        input_layout.addWidget(self.btn_send)
        
        layout.addWidget(input_frame)

        return page

    def create_general_chat_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # --- Header ---
        header_layout = QHBoxLayout()
        
        self.btn_toggle_gen = QPushButton("≡")
        self.btn_toggle_gen.setObjectName("MenuButton")
        self.btn_toggle_gen.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_gen.setFixedSize(40, 40)
        self.btn_toggle_gen.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.btn_toggle_gen)

        header = QLabel("محادثة عامة مع الذكاء الاصطناعي")
        header.setObjectName("HeaderTitle")
        header_layout.addWidget(header)
        header_layout.addStretch()

        self.btn_clear_general = QPushButton("مسح المحادثة")
        self.btn_clear_general.setObjectName("ClearButton")
        self.btn_clear_general.setCursor(Qt.PointingHandCursor)
        self.btn_clear_general.clicked.connect(self.clear_general_chat)
        header_layout.addWidget(self.btn_clear_general)

        layout.addLayout(header_layout)

        # --- Chat Area ---
        self.general_chat_scroll = QScrollArea()
        self.general_chat_scroll.setWidgetResizable(True)
        self.general_chat_scroll.setObjectName("ChatScroll")
        self.general_chat_scroll.setFrameShape(QFrame.NoFrame) # إزالة الإطار
        self.general_chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # الوعاء الداخلي
        self.general_chat_widget = QWidget()
        # هذا السطر هو الأهم: جعل الخلفية شفافة لتأخذ لون التطبيق الأصلي
        self.general_chat_widget.setStyleSheet("background: transparent;") 
        
        self.general_chat_layout = QVBoxLayout(self.general_chat_widget)
        self.general_chat_layout.setSpacing(15)
        self.general_chat_layout.setContentsMargins(10, 10, 10, 10)
        self.general_chat_layout.setAlignment(Qt.AlignTop)
        
        self.general_chat_scroll.setWidget(self.general_chat_widget)
        layout.addWidget(self.general_chat_scroll, stretch=1)

        # --- Input Area ---
        # استخدام نفس ID الإطار (InputFrame) ليأخذ نفس الستايل من ملف CSS
        input_frame = QFrame()
        input_frame.setObjectName("InputFrame") 
        input_frame.setFixedHeight(80)

        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.general_txt_input = QLineEdit()
        self.general_txt_input.setPlaceholderText("اكتب رسالتك هنا...")
        self.general_txt_input.returnPressed.connect(self.start_general_chat_thread)
        self.general_txt_input.setLayoutDirection(Qt.RightToLeft)
        self.general_txt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.general_btn_send = QPushButton("إرسال")
        self.general_btn_send.setObjectName("SendButton")
        self.general_btn_send.setCursor(Qt.PointingHandCursor)
        self.general_btn_send.setMinimumHeight(45)
        self.general_btn_send.clicked.connect(self.on_general_send_button_clicked)

        input_layout.addWidget(self.general_txt_input)
        input_layout.addWidget(self.general_btn_send)
        layout.addWidget(input_frame)

        return page

    def add_bubble(self, text, bubble_type="user"):
        # 1. حاوية الصف
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 5, 10, 5)
        row_layout.setSpacing(10)

        # 2. إنشاء الفقاعة
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # ضبط اتجاه النص داخل الفقاعة لليمين دائماً (للعربية)
        bubble.setLayoutDirection(Qt.RightToLeft)

        # 3. حساب العرض (لمنع التكدس العمودي)
        viewport_width = self.chat_scroll.viewport().width() if self.chat_scroll.viewport().width() > 0 else 600
        max_bubble_width = int(viewport_width * 0.80)
        
        font_metrics = bubble.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        
        # عرض الفقاعة بناءً على طول النص
        bubble_width = min(max(text_width + 40, 100), max_bubble_width)
        bubble.setFixedWidth(bubble_width)

        # الستايل العام للخطوط
        common_style = """
            QLabel {
                font-family: "Segoe UI", "Arial", sans-serif;
                font-size: 16px;
                padding: 10px 15px;
                border-radius: 15px;
                line-height: 1.5;
            }
        """

        # --- التعامل مع الاتجاهات (بناءً على أن التطبيق RTL) ---

        if bubble_type == "user":
            # المستخدم: نريدها في اليمين
            # بما أن التخطيط عربي (يبدأ من اليمين)، نضيف الفقاعة أولاً
            
            bubble.setStyleSheet(common_style + f"""
                QLabel {{
                    background-color: {self.colors['accent']};
                    color: {self.colors['bg_dark']};
                    border-bottom-right-radius: 2px;
                }}
            """)
            bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # الترتيب: الفقاعة (يمين) -> فراغ (يدفع لليسار)
            row_layout.addWidget(bubble)
            row_layout.addStretch()

        elif bubble_type == "bot":
            # البوت: نريدها في اليسار
            # نضيف الفراغ أولاً ليدفع الفقاعة لليسار
            
            bubble.setStyleSheet(common_style + f"""
                QLabel {{
                    background-color: {self.colors['bg_card']};
                    color: white;
                    border: 1px solid {self.colors['accent']};
                    border-bottom-left-radius: 2px;
                }}
            """)
            bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # الترتيب: فراغ -> فقاعة (يسار)
            row_layout.addStretch()
            row_layout.addWidget(bubble)

        # --- تعديل رسائل النظام والتحميل (اللون والحجم) ---
        # ... (بعد شروط user و bot) ...

        elif bubble_type == "loading" or bubble_type == "system":
             # 1. إلغاء العرض الثابت الذي وضعناه في بداية الدالة
             # نسمح للفقاعة أن تأخذ حجم النص فقط لتكون صغيرة ومرتبة
             bubble.setMinimumWidth(0)
             bubble.setMaximumWidth(16777215) # إلغاء الحد الأقصى السابق (QWIDGETSIZE_MAX)
             
             # 2. تصميم "نظامي" شفاف وهادئ
             bubble.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.1); /* خلفية سوداء شفافة  */
                    color: rgba(255, 255, 255, 0.5);      /* نص أبيض نصف شفاف (رمادي فاتح) */
                    font-size: 12px;                      /* خط صغير */
                    font-family: "Segoe UI", sans-serif;
                    font-style: italic;                   /* خط مائل للدلالة على أنها عملية خلفية */
                    padding: 4px 12px;                    /* حشوة صغيرة */
                    border-radius: 10px;                  /* حواف دائرية صغيرة (كأنها شارة) */
                    
                }
             """)
             
             # 3. المحاذاة في المنتصف تماماً
             bubble.setAlignment(Qt.AlignCenter)
             
             # الترتيب: فراغ -> الرسالة -> فراغ (لتظهر في الوسط)
             row_layout.addStretch()
             row_layout.addWidget(bubble)
             row_layout.addStretch()

        
        elif bubble_type == "error":
             bubble.setFixedWidth(0)
             bubble.setMaximumWidth(max_bubble_width)
             bubble.setStyleSheet("""
                QLabel {
                    color: #ffcccc;
                    background-color: rgba(255, 0, 0, 0.1);
                    border: 1px solid #ff4444;
                    font-size: 14px;
                    border-radius: 10px;
                    padding: 8px 15px;
                }
             """)
             bubble.setAlignment(Qt.AlignCenter)
             row_layout.addStretch()
             row_layout.addWidget(bubble)
             row_layout.addStretch()

        # إضافة الصف
        self.chat_layout.addWidget(row_widget)
        
        # التمرير للأسفل
        QApplication.processEvents()
        QTimer.singleShot(10, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))

    def on_send_button_clicked(self):
        # Check if there's an active worker
        if hasattr(self, 'worker') and self.worker.isRunning():
            # Cancel the current operation
            self.cancel_current_operation()
        else:
            # Start a new chat
            self.start_chat_thread()

    def cancel_current_operation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            print("Cancelling current operation...")
            self.worker.terminate()
            self.worker.wait(1000)
            
            # إزالة مؤشر التحميل
            if self.chat_layout.count() > 0:
                last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
                if last_item and last_item.layout():
                    for i in range(last_item.layout().count()):
                        widget = last_item.layout().itemAt(i).widget()
                        if widget and isinstance(widget, QLabel) and "جاري معالجة" in widget.text():
                            while last_item.layout().count():
                                item = last_item.layout().takeAt(0)
                                if item.widget():
                                    item.widget().deleteLater()
                            self.chat_layout.removeItem(last_item)
                            break
            
            # إضافة رسالة الإلغاء
            self.add_bubble("تم إلغاء العملية", "loading")
            
            # Re-enable button and stop timer
            self.btn_send.setEnabled(True)
            self.btn_send.setText("إرسال الطلب")
            if hasattr(self, 'worker_timer') and self.worker_timer.isActive():
                self.worker_timer.stop()

    def create_settings_page(self):
        page = QWidget()
        # استخدام سكرول في حال كانت الشاشة صغيرة
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 40, 50, 40)
        layout.setAlignment(Qt.AlignTop)

        header = QLabel("تهيئة النظام والاتصال")
        header.setObjectName("HeaderTitle")
        layout.addWidget(header)
        layout.addSpacing(10)

        # Card Container for Settings
        settings_card = QFrame()
        settings_card.setObjectName("SettingsCard")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)

        # Fields Container
        fields_container = QWidget()
        fields_layout = QVBoxLayout(fields_container)
        fields_layout.setSpacing(6)
        fields_layout.setContentsMargins(0, 0, 0, 0)

        # Database Connection Section
        db_section = QLabel("🔗 إعدادات قاعدة البيانات")
        db_section.setStyleSheet(f"font-weight: bold; color: {self.colors['accent']}; font-size: 14px; margin-bottom: 5px;")
        fields_layout.addWidget(db_section)

        # Host Field
        host_container = QWidget()
        host_layout = QHBoxLayout(host_container)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)
        host_label = QLabel("عنوان السيرفر:")
        host_label.setObjectName("SettingsLabel")
        host_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_host = QLineEdit()
        self.input_host.setObjectName("SettingsInput")
        self.input_host.setPlaceholderText("مثال: localhost")
        self.input_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        host_layout.addWidget(host_label)
        host_layout.addWidget(self.input_host, stretch=1)
        fields_layout.addWidget(host_container)

        # Port Field
        port_container = QWidget()
        port_layout = QHBoxLayout(port_container)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(6)
        port_label = QLabel("المنفذ:")
        port_label.setObjectName("SettingsLabel")
        port_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_port = QLineEdit()
        self.input_port.setObjectName("SettingsInput")
        self.input_port.setPlaceholderText("مثال: 5432")
        self.input_port.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.input_port, stretch=1)
        fields_layout.addWidget(port_container)

        # Database Name Field
        db_container = QWidget()
        db_layout = QHBoxLayout(db_container)
        db_layout.setContentsMargins(0, 0, 0, 0)
        db_layout.setSpacing(6)
        db_label = QLabel("اسم قاعدة البيانات:")
        db_label.setObjectName("SettingsLabel")
        db_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_db = QLineEdit()
        self.input_db.setObjectName("SettingsInput")
        self.input_db.setPlaceholderText("مثال: company_db")
        self.input_db.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        db_layout.addWidget(db_label)
        db_layout.addWidget(self.input_db, stretch=1)
        fields_layout.addWidget(db_container)

        # User Field
        user_container = QWidget()
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(0, 0, 0, 0)
        user_layout.setSpacing(6)
        user_label = QLabel("المستخدم:")
        user_label.setObjectName("SettingsLabel")
        user_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_user = QLineEdit()
        self.input_user.setObjectName("SettingsInput")
        self.input_user.setPlaceholderText("مثال: postgres")
        self.input_user.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        user_layout.addWidget(user_label)
        user_layout.addWidget(self.input_user, stretch=1)
        fields_layout.addWidget(user_container)

        # Password Field
        pass_container = QWidget()
        pass_layout = QHBoxLayout(pass_container)
        pass_layout.setContentsMargins(0, 0, 0, 0)
        pass_layout.setSpacing(6)
        pass_label = QLabel("كلمة المرور:")
        pass_label.setObjectName("SettingsLabel")
        pass_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_pass = QLineEdit()
        self.input_pass.setObjectName("SettingsInput")
        self.input_pass.setEchoMode(QLineEdit.Password)
        self.input_pass.setPlaceholderText("أدخل كلمة المرور")
        self.input_pass.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pass_layout.addWidget(pass_label)
        pass_layout.addWidget(self.input_pass, stretch=1)
        fields_layout.addWidget(pass_container)

        # AI Model Section
        fields_layout.addSpacing(8)
        ai_section = QLabel("🤖 نموذج الذكاء الاصطناعي")
        ai_section.setStyleSheet(f"font-weight: bold; color: {self.colors['accent']}; font-size: 14px; margin-bottom: 5px;")
        fields_layout.addWidget(ai_section)

        # Model Field
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(6)
        model_label = QLabel("نموذج AI:")
        model_label.setObjectName("SettingsLabel")
        model_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.combo_model = QComboBox()
        self.combo_model.setObjectName("SettingsCombo")
        self.combo_model.setMinimumHeight(28)
        self.combo_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.combo_model, stretch=1)
        fields_layout.addWidget(model_container)

        settings_layout.addWidget(fields_container)

        layout.addWidget(settings_card)
        layout.addSpacing(8)

        # Save Button
        btn_save = QPushButton("حفظ التغييرات وإعادة التشغيل")
        btn_save.setObjectName("SaveButton")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save, alignment=Qt.AlignCenter)
        
        layout.addStretch()
        return page

    def create_history_page(self):
        """إنشاء صفحة المحفوظات"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("سجل المحادثات السابقة")
        header.setObjectName("HeaderTitle")
        header.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {self.colors['accent']};")
        layout.addWidget(header)
        
        # Store reference to scroll area for refresh
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {self.colors['bg_main']};
            }}
        """)
        
        # Content will be filled by refresh_history_page
        self.history_content_widget = QWidget()
        self.history_content_layout = QVBoxLayout(self.history_content_widget)
        self.history_scroll.setWidget(self.history_content_widget)
        
        layout.addWidget(self.history_scroll)
        layout.addStretch()
        
        return page
    
    def refresh_history_page(self):
        """تحديث محتوى صفحة المحفوظات"""
        # مسح المحتوى القديم
        while self.history_content_layout.count():
            item = self.history_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Load conversations
        conversations = self.history_manager.load_all_conversations()
        
        if not conversations:
            empty_label = QLabel("لا توجد محادثات مسجلة بعد")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #AAAAAA; font-size: 14px;")
            self.history_content_layout.addWidget(empty_label)
        else:
            for conv in reversed(conversations):  # عرض الأحدث أولاً
                conv_frame = QFrame()
                conv_frame.setObjectName("SettingsCard")
                conv_layout = QVBoxLayout(conv_frame)
                conv_layout.setContentsMargins(15, 12, 15, 12)
                
                # Title and timestamp
                header_layout = QHBoxLayout()
                title = QLabel(conv.get('title', 'محادثة بدون عنوان'))
                title.setStyleSheet(f"font-weight: bold; color: {self.colors['accent']}; font-size: 13px;")
                timestamp = QLabel(conv.get('timestamp', ''))
                timestamp.setStyleSheet(f"color: {self.colors['text_dim']}; font-size: 10px;")
                timestamp.setAlignment(Qt.AlignLeft)
                
                header_layout.addWidget(title)
                header_layout.addStretch()
                header_layout.addWidget(timestamp)
                conv_layout.addLayout(header_layout)
                
                # Message preview
                if conv.get('messages'):
                    preview = conv.get('messages', [{}])[0].get('content', '')
                    if len(preview) > 80:
                        preview = preview[:80] + "..."
                    preview_label = QLabel(preview)
                    preview_label.setStyleSheet(f"color: {self.colors['text_dim']}; font-size: 11px;")
                    preview_label.setWordWrap(True)
                    conv_layout.addWidget(preview_label)
                
                # Buttons layout
                buttons_layout = QHBoxLayout()
                
                # Load button
                load_btn = QPushButton("تحميل")
                load_btn.setObjectName("SaveButton")
                load_btn.setCursor(Qt.PointingHandCursor)
                load_btn.setMaximumWidth(100)
                load_btn.clicked.connect(lambda checked, c=conv: self.load_conversation(c))
                
                # Delete button
                delete_btn = QPushButton("حذف")
                delete_btn.setObjectName("SaveButton")
                delete_btn.setCursor(Qt.PointingHandCursor)
                delete_btn.setMaximumWidth(100)
                delete_btn.setStyleSheet(f"background-color: #c74444;")
                delete_btn.clicked.connect(lambda checked, c=conv: self.delete_conversation_item(c))
                
                buttons_layout.addStretch()
                buttons_layout.addWidget(load_btn)
                buttons_layout.addWidget(delete_btn)
                conv_layout.addLayout(buttons_layout)
                
                self.history_content_layout.addWidget(conv_frame)
        
        self.history_content_layout.addStretch()

    def trigger_ollama_load(self):
        # هذه الدالة تستدعى لتشغيل الموديل في الخلفية
        model = self.combo_model.currentText()
        
        # Check if Ollama is available first
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.add_bubble("⚠️ تحذير: Ollama غير متاح. تأكد من تشغيله أولاً.", "error")
                return
        except Exception as e:
            self.add_bubble(f"⚠️ تحذير: فشل التحقق من Ollama: {str(e)}", "error")
            return
        
        self.ollama_loader = OllamaLoaderThread(model)
        self.ollama_loader.finished_signal.connect(self.on_ollama_ready)
        self.ollama_loader.start()
        
        # رسالة في الشات تخبر المستخدم
        welcome_msg = """مرحباً بك في نظام التحليل الذكي
• يجب وجود قاعدة بيانات PostgreSQL ووضع بياناتها في الإعدادات"""
        self.add_bubble(welcome_msg, "system")
        self.add_bubble(f"جاري تهيئة الموديل {model}...", "loading")

    def get_available_models(self):
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # First line is header
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]  # Take full name with tag
                            if model_name not in models:
                                models.append(model_name)
                    return models if models else ["llama3"]
            return ["llama3"]  # Default if failed
        except Exception as e:
            print(f"Error getting models: {e}")
            return ["llama3"]

    def on_ollama_ready(self, msg):
        self.add_bubble(f"✅ {msg}", "success")
        
        # Now that the model is ready, generate suggestions
        current_settings = {
            "host": self.input_host.text(),
            "port": self.input_port.text(),
            "dbname": self.input_db.text(),
            "user": self.input_user.text(),
            "pass": self.input_pass.text(),
            "model": self.combo_model.currentText()
        }
        self.suggestion_worker = SuggestionWorker(current_settings)
        self.suggestion_worker.suggestions_ready.connect(self.on_suggestions_ready)
        self.suggestion_worker.start()

    def on_suggestions_ready(self, suggestions):
        # Clear old suggestions
        for i in reversed(range(self.suggestions_layout.count())): 
            self.suggestions_layout.itemAt(i).widget().setParent(None)

        # Add new suggestion buttons
        for suggestion_text in suggestions[:4]: # Limit to 4 suggestions
            btn = SuggestionButton(suggestion_text, self.colors['accent'], self.colors['bg_card'])
            btn.clicked.connect(lambda checked, text=suggestion_text: self.on_suggestion_clicked(text))
            self.suggestions_layout.addWidget(btn)

    def on_suggestion_clicked(self, text):
        self.welcome_input.setText(text)
        self.transition_to_chat()

    def save_settings(self):
        data = {
            "host": self.input_host.text(),
            "port": self.input_port.text(),
            "dbname": self.input_db.text(),
            "user": self.input_user.text(),
            "pass": self.input_pass.text(),
            "model": self.combo_model.currentText()
        }
        with open(self.settings_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        QMessageBox.information(self, "نجاح", "تم حفظ الإعدادات. سيتم إعادة تهيئة الموديل الآن.")
        self.trigger_ollama_load()

    def ensure_config_exists(self):
        """تأكد من وجود ملف الإعدادات وإنشاؤه بالقيم الافتراضية إن لم يكن موجوداً"""
        if not os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                print(f"تم إنشاء ملف الإعدادات: {self.settings_file}")
            except Exception as e:
                print(f"تحذير: فشل إنشاء ملف الإعدادات: {e}")

    def load_settings(self):
        # Set default values first using centralized defaults
        self.input_host.setText(DEFAULT_CONFIG["host"])
        self.input_port.setText(DEFAULT_CONFIG["port"])
        self.input_db.setText(DEFAULT_CONFIG["dbname"])
        self.input_user.setText(DEFAULT_CONFIG["user"])
        self.input_pass.setText(DEFAULT_CONFIG["pass"])
        default_model = DEFAULT_CONFIG["model"]
        
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                    self.input_host.setText(data.get("host", DEFAULT_CONFIG["host"]))
                    self.input_port.setText(data.get("port", DEFAULT_CONFIG["port"]))
                    self.input_db.setText(data.get("dbname", DEFAULT_CONFIG["dbname"]))
                    self.input_user.setText(data.get("user", DEFAULT_CONFIG["user"]))
                    self.input_pass.setText(data.get("pass", DEFAULT_CONFIG["pass"]))
                    default_model = data.get("model", DEFAULT_CONFIG["model"])
            except Exception as e:
                print(f"تحذير: فشل تحميل الإعدادات: {e}") # Keep default values if config is corrupted
        
        # Update model combo with available models
        available_models = self.get_available_models()
        self.combo_model.clear()
        self.combo_model.addItems(available_models)
        
        # Set the saved model if available, otherwise first one
        if default_model in available_models:
            self.combo_model.setCurrentText(default_model)
        elif available_models:
            self.combo_model.setCurrentText(available_models[0])

    def clear_chat(self):
        reply = QMessageBox.question(self, 'مسح المحادثة', 
                                     "هل أنت متأكد أنك تريد مسح المحادثة؟",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # مسح جميع الفقاعات
            while self.chat_layout.count():
                item = self.chat_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    while item.layout().count():
                        sub_item = item.layout().takeAt(0)
                        if sub_item.widget():
                            sub_item.widget().deleteLater()
            
            # إضافة رسالة تأكيد
            self.add_bubble("✅ النظام جاهز.", "success")

    def start_chat_thread(self):
        question = self.txt_input.text()
        if not question: return

        settings = {
            "host": self.input_host.text(),
            "port": self.input_port.text(),
            "dbname": self.input_db.text(),
            "user": self.input_user.text(),
            "pass": self.input_pass.text(),
            "model": self.combo_model.currentText()
        }

        # إضافة فقاعة السؤال
        self.add_bubble(question, "user")
        self.txt_input.clear()
        
        # إضافة مؤشر التحميل
        self.add_bubble("جاري معالجة الاستعلام...", "loading")
        self.btn_send.setText("إلغاء")  # Change text to cancel
        
        # Scroll to bottom
        sb = self.chat_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

        try:
            self.worker = ChatWorker(settings, question)
            self.worker.response_received.connect(self.handle_response)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.finished.connect(self.on_worker_finished)  # Add finished signal
            self.worker.start()
            
            # Add timeout to prevent hanging
            self.worker_timer = QTimer()
            self.worker_timer.setSingleShot(True)
            self.worker_timer.timeout.connect(self.on_worker_timeout)
            self.worker_timer.start(600000)  # 10 minute timeout
            
        except Exception as e:
            self.handle_error(f"خطأ في بدء المعالجة: {str(e)}")

    def on_worker_finished(self):
        # Ensure button is re-enabled when worker finishes
        self.btn_send.setEnabled(True)
        self.btn_send.setText("إرسال الطلب")  # Reset button text
        # Stop and clean up timer if it exists
        if hasattr(self, 'worker_timer') and self.worker_timer.isActive():
            self.worker_timer.stop()

    def on_worker_timeout(self):
        # Check if worker is still running and terminate if needed
        if hasattr(self, 'worker') and self.worker.isRunning():
            print("Worker timeout - terminating thread")
            self.worker.terminate()
            self.worker.wait(1000)  # Wait up to 1 second for termination
            self.handle_error("انتهت مهلة المعالجة (45 ثانية). يرجى المحاولة مرة أخرى أو التحقق من اتصال Ollama.")
        
        # Reset button text
        self.btn_send.setText("إرسال الطلب")
        
        # Clean up timer
        if hasattr(self, 'worker_timer'):
            self.worker_timer.stop()

    def handle_response(self, response):
        # إزالة مؤشر التحميل (الفقاعة الأخيرة إذا كانت loading)
        if self.chat_layout.count() > 0:
            # البحث عن آخر فقاعة وإزالتها إذا كانت loading
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if last_item and last_item.layout():
                # تحقق من وجود label في آخر layout
                for i in range(last_item.layout().count()):
                    widget = last_item.layout().itemAt(i).widget()
                    if widget and isinstance(widget, QLabel) and "جاري معالجة" in widget.text():
                        # إزالة الفقاعة
                        while last_item.layout().count():
                            item = last_item.layout().takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                        self.chat_layout.removeItem(last_item)
                        break

        # إضافة فقاعة الإجابة
        self.add_bubble(response, "bot")
        self.btn_send.setEnabled(True)
        self.btn_send.setText("إرسال الطلب")  # Reset button text

    def handle_error(self, error_msg):
        # إزالة مؤشر التحميل
        if self.chat_layout.count() > 0:
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if last_item and last_item.layout():
                for i in range(last_item.layout().count()):
                    widget = last_item.layout().itemAt(i).widget()
                    if widget and isinstance(widget, QLabel) and "جاري معالجة" in widget.text():
                        while last_item.layout().count():
                            item = last_item.layout().takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                        self.chat_layout.removeItem(last_item)
                        break

        # إضافة فقاعة الخطأ
        error_text = f"خطأ في النظام: {error_msg}\n\nنصائح للحل:\n• تأكد من تشغيل Ollama: ollama serve\n• تأكد من تحميل النموذج: ollama pull llama3\n• تحقق من إعدادات قاعدة البيانات\n• تأكد من اتصال الإنترنت"
        self.add_bubble(error_text, "error")
        self.btn_send.setEnabled(True)
        self.btn_send.setText("إرسال الطلب")  # Reset button text

    def clear_general_chat(self):
        reply = QMessageBox.question(self, 'مسح المحادثة', 
                                     "هل أنت متأكد أنك تريد مسح المحادثة العامة؟",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # مسح جميع الفقاعات
            while self.general_chat_layout.count():
                item = self.general_chat_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # مسح التاريخ
            self.general_chat_history = []
            
            # إضافة رسالة تأكيد
            self.add_general_bubble("✅ تم مسح المحادثة.", "success")

    def start_general_chat_thread(self):
        # أخذ النص من حقل المحادثة العامة
        question = self.general_txt_input.text()
        if not question: return

        # 1. إضافة فقاعة المستخدم في الشات العام
        self.add_general_bubble(question, "user")
        
        # 2. حفظ في التاريخ الخاص بالشات العام
        self.general_chat_history.append({"role": "user", "content": question})
        
        # 3. تنظيف الحقل
        self.general_txt_input.clear()
        
        # 4. إضافة مؤشر تحميل في الشات العام
        self.add_general_bubble("جاري التفكير...", "loading")
        self.general_btn_send.setText("إلغاء")
        
        try:
            settings = {
                "model": self.combo_model.currentText()
            }
            # إرسال تاريخ المحادثة العامة فقط
            self.general_worker = GeneralChatWorker(settings, question, self.general_chat_history[:-1])
            self.general_worker.response_received.connect(self.handle_general_response)
            self.general_worker.error_occurred.connect(self.handle_general_error)
            self.general_worker.finished.connect(self.on_general_worker_finished)
            self.general_worker.start()
            
            # Timer
            self.general_worker_timer = QTimer()
            self.general_worker_timer.setSingleShot(True)
            self.general_worker_timer.timeout.connect(self.on_general_worker_timeout)
            self.general_worker_timer.start(45000)
            
        except Exception as e:
            self.handle_general_error(f"خطأ في بدء المحادثة: {str(e)}")

    def on_general_send_button_clicked(self):
        if hasattr(self, 'general_worker') and self.general_worker.isRunning():
            self.cancel_general_operation()
        else:
            self.start_general_chat_thread()

    def cancel_general_operation(self):
        if hasattr(self, 'general_worker') and self.general_worker.isRunning():
            print("Cancelling general operation...")
            self.general_worker.terminate()
            self.general_worker.wait(1000)
            
            # إزالة مؤشر التحميل
            if self.general_chat_layout.count() > 0:
                last_item = self.general_chat_layout.itemAt(self.general_chat_layout.count() - 1)
                if last_item and last_item.layout():
                    for i in range(last_item.layout().count()):
                        widget = last_item.layout().itemAt(i).widget()
                        if widget and isinstance(widget, QLabel) and "جاري التفكير" in widget.text():
                            while last_item.layout().count():
                                item = last_item.layout().takeAt(0)
                                if item.widget():
                                    item.widget().deleteLater()
                            self.general_chat_layout.removeItem(last_item)
                            break
            
            # إضافة رسالة الإلغاء
            self.add_general_bubble("تم إلغاء العملية", "loading")
            
            self.general_btn_send.setEnabled(True)
            self.general_btn_send.setText("إرسال")
            if hasattr(self, 'general_worker_timer') and self.general_worker_timer.isActive():
                self.general_worker_timer.stop()

    def on_general_worker_finished(self):
        self.general_btn_send.setEnabled(True)
        self.general_btn_send.setText("إرسال")
        if hasattr(self, 'general_worker_timer') and self.general_worker_timer.isActive():
            self.general_worker_timer.stop()

    def on_general_worker_timeout(self):
        if hasattr(self, 'general_worker') and self.general_worker.isRunning():
            print("General worker timeout - terminating thread")
            self.general_worker.terminate()
            self.general_worker.wait(1000)
            self.handle_general_error("انتهت مهلة المعالجة (45 ثانية). يرجى المحاولة مرة أخرى.")
        
        self.general_btn_send.setText("إرسال")
        
        if hasattr(self, 'general_worker_timer'):
            self.general_worker_timer.stop()

    def handle_general_response(self, response):
        # 1. إزالة مؤشر التحميل من الشات العام
        if self.general_chat_layout.count() > 0:
            last_item = self.general_chat_layout.itemAt(self.general_chat_layout.count() - 1)
            if last_item and last_item.layout():
                for i in range(last_item.layout().count()):
                    widget = last_item.layout().itemAt(i).widget()
                    # نتأكد أننا نحذف فقاعة التحميل فقط
                    if widget and isinstance(widget, QLabel) and "جاري التفكير" in widget.text():
                        while last_item.layout().count():
                            item = last_item.layout().takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                        self.general_chat_layout.removeItem(last_item)
                        break

        # 2. إضافة رد البوت للشات العام
        self.add_general_bubble(response, "bot")
        
        # 3. حفظ في التاريخ وحفظ المحادثة الحالية
        self.general_chat_history.append({"role": "assistant", "content": response})
        
        # حفظ تلقائي للمحادثة الحالية
        if len(self.general_chat_history) > 0 and len(self.general_chat_history) % 2 == 0:  # حفظ كل زوج من الرسائل
            first_user_msg = next((msg['content'][:50] for msg in self.general_chat_history if msg['role'] == 'user'), "محادثة عامة")
            self.save_current_conversation(f"محادثة: {first_user_msg}")
        
        # 4. إعادة تفعيل الزر
        self.general_btn_send.setEnabled(True)
        self.general_btn_send.setText("إرسال")

    def handle_general_error(self, error_msg):
        # إزالة مؤشر التحميل من الشات العام
        if self.general_chat_layout.count() > 0:
            last_item = self.general_chat_layout.itemAt(self.general_chat_layout.count() - 1)
            if last_item and last_item.layout():
                # ... (نفس منطق الإزالة أعلاه) ...
                for i in range(last_item.layout().count()):
                    widget = last_item.layout().itemAt(i).widget()
                    if widget and isinstance(widget, QLabel) and "جاري التفكير" in widget.text():
                        while last_item.layout().count():
                            item = last_item.layout().takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                        self.general_chat_layout.removeItem(last_item)
                        break

        # إضافة فقاعة الخطأ في الشات العام
        self.add_general_bubble(f"⚠️ {error_msg}", "error")
        self.general_btn_send.setEnabled(True)
        self.general_btn_send.setText("إرسال")

    def apply_styles(self):
        # ملف CSS شامل ومحسن
        css = f"""
    /* --- Menu Toggle Button --- */
        QPushButton#MenuButton {{
            background-color: transparent;
            color: {self.colors['accent']};
            border: 1px solid {self.colors['accent']};
            border-radius: 8px;
            font-size: 24px;
            padding-bottom: 5px; /* رفع الرمز قليلاً للأعلى */
        }}
        QPushButton#MenuButton:hover {{
            background-color: {self.colors['accent']};
            color: {self.colors['bg_dark']};
        }}

        QMainWindow {{
            background-color: {self.colors['bg_main']};
        }}
        /* --- Sidebar Styles --- */
        QFrame#Sidebar {{
            background-color: {self.colors['bg_dark']};
            border-right: 1px solid {self.colors['accent']};
        }}
        QLabel#AppTitle {{
            color: {self.colors['accent']};
            font-size: 20px;
            font-weight: bold;
            letter-spacing: 1px;
        }}
        QLabel#Footer {{
            color: #666;
            font-size: 10px;
        }}
        QLabel#LogoPlaceholder {{
            color: {self.colors['accent']};
            border: 2px dashed {self.colors['accent']};
            padding: 20px;
            font-weight: bold;
        }}
        
        /* --- Navigation Buttons --- */
        QPushButton#NavButton {{
            background-color: transparent;
            color: #ccc;
            border: none;
            text-align: left;
            padding: 15px 25px;
            font-size: 16px;
            border-left: 4px solid transparent;
        }}
        QPushButton#NavButton:hover {{
            background-color: rgba(255, 255, 255, 0.05);
            color: white;
        }}
        QPushButton#NavButton:checked {{
            background-color: rgba(197, 160, 89, 0.1); /* لون ذهبي شفاف */
            color: {self.colors['accent']};
            border-left: 4px solid {self.colors['accent']};
            font-weight: bold;
        }}

        /* --- Content Area --- */
        QFrame#ContentArea {{
            background-color: {self.colors['bg_main']};
        }}
        QLabel#HeaderTitle {{
            color: {self.colors['accent']};
            font-size: 28px;
            font-weight: bold;
            border-bottom: 2px solid {self.colors['accent']};
            padding-bottom: 15px;
        }}

        /* --- Settings Card & Inputs --- */
        QFrame#SettingsCard {{
            background-color: {self.colors['bg_card']};
            border-radius: 15px;
            border: 1px solid #444;
        }}
        QLabel#SettingsLabel {{
            color: {self.colors['text']};
            font-size: 10px;
            font-weight: bold;
            min-width: 75px;
            text-align: right;
            padding-right: 4px;
        }}
        QLineEdit#SettingsInput, QComboBox#SettingsCombo {{
            background-color: {self.colors['bg_dark']};
            color: white;
            border: 1px solid #555;
            border-radius: 6px;
            padding: 5px 8px;
            font-size: 10px;
            font-family: "Segoe UI", "Arial", sans-serif;
            min-height: 28px;
            selection-background-color: {self.colors['accent']};
            selection-color: black;
        }}
        QLineEdit#SettingsInput:focus, QComboBox#SettingsCombo:focus {{
            border: 2px solid {self.colors['accent']};
        }}
        QComboBox#SettingsCombo {{
            padding-right: 20px; /* Space for dropdown arrow */
        }}
        QComboBox#SettingsCombo::drop-down {{
            border: none;
            background: transparent;
            width: 20px;
        }}
        QComboBox#SettingsCombo::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid {self.colors['text']};
            margin-right: 5px;
        }}
        QComboBox#SettingsCombo QAbstractItemView {{
            background-color: {self.colors['bg_dark']};
            color: white;
            border: 1px solid {self.colors['accent']};
            border-radius: 4px;
            selection-background-color: {self.colors['accent']};
            selection-color: black;
        }}

        /* --- Chat Area --- */
        QTextEdit#ChatDisplay {{
            background-color: {self.colors['bg_card']};
            color: #eee;
            border: none;
            border-radius: 15px;
            padding: 25px;
            font-size: 18px;
            line-height: 1.6;
            font-family: "itf Qomra Arabic", Arial, sans-serif;
        }}
        QFrame#InputFrame {{
            background-color: {self.colors['bg_card']};
            border-radius: 30px;
            border: 1px solid #555;
            padding: 5px;
        }}
        QFrame#InputFrame QLineEdit {{
            background-color: transparent;
            border: none;
            font-size: 16px;
            color: white;
            border-radius: 25px;
            padding: 10px 20px;
            min-height: 45px;
            text-align: right;
        }}
        QFrame#InputFrame QLineEdit:focus {{
            border: none;
        }}

        QPushButton#SendButton, QPushButton#SaveButton {{
            background-color: {self.colors['accent']};
            color: #002b28;
            font-weight: bold;
            font-size: 11px;
            border-radius: 14px;
            padding: 5px 16px;
            border: 2px solid {self.colors['accent']};
            min-width: 140px;
            max-width: 180px;
        }}
        QPushButton#SendButton:hover, QPushButton#SaveButton:hover {{
            background-color: {self.colors['accent_hover']};
            border-color: {self.colors['accent_hover']};
        }}
        QPushButton#SendButton:pressed {{
            background-color: {self.colors['bg_dark']};
            color: {self.colors['accent']};
        }}
        QPushButton#ClearButton {{
            background-color: transparent;
            color: {self.colors['text_dim']};
            border: 1px solid {self.colors['bg_card']};
            font-weight: bold;
            font-size: 13px;
            border-radius: 15px;
            padding: 8px 20px;
            max-width: 120px;
        }}
        QPushButton#ClearButton:hover {{
            background-color: {self.colors['bg_card']};
            color: white;
        }}
        
        /* --- Welcome Page Styles --- */
        QLabel#WelcomeTitle {{
            font-size: 48px;
            font-weight: bold;
            color: {self.colors['accent']};
        }}
        QLabel#WelcomeSubtitle {{
            font-size: 16px;
            color: {self.colors['text_dim']};
            margin-bottom: 20px;
        }}
        QFrame#WelcomeInputFrame {{
            max-width: 80%;
            min-width: 500px; /* ضمان عرض جيد للإطار نفسه */
            background-color: {self.colors['bg_card']};
            border-radius: 35px;
            border: 1px solid {self.colors['accent']};
            padding: 5px;

        }}
        QLineEdit#WelcomeInput {{
            background: transparent;
            border: none;
            color: white;
            font-size: 18px;
            padding: 15px 15px;
            min-height: 50px;
        }}
        QPushButton#WelcomeSendButton {{
            background-color: {self.colors['accent']};
            color: {self.colors['bg_dark']};
            font-weight: bold;
            font-size: 16px;
            border-radius: 25px;
            padding: 12px 35px;
            border: 2px solid {self.colors['accent']};
        }}
        QPushButton#WelcomeSendButton:hover {{
            background-color: {self.colors['accent_hover']};
            border-color: {self.colors['accent_hover']};
        }}
        QPushButton#WelcomeSendButton:pressed {{
            background-color: {self.colors['bg_dark']};
            color: {self.colors['accent']};
        }}

        /* --- Chat Bubbles --- */
        .chat-bubble {{
            margin-bottom: 15px;
            font-family: "itf Qomra Arabic", Arial, sans-serif;
        }}
        .chat-bubble div {{
            
            word-wrap: break-word;
        }}
        """
        self.setStyleSheet(css)
    
    def load_conversation(self, conversation):
        """تحميل محادثة سابقة"""
        try:
            # تحميل المحادثة في الواجهة
            if conversation.get('type') == 'general_chat' and conversation.get('messages'):
                # تحميل محادثة عامة
                self.btn_general.click()  # الذهاب لصفحة المحادثة العامة
                self.general_chat_history = conversation.get('messages', [])
                
                # إعادة عرض المحادثة
                QApplication.processEvents()
                
                # مسح المحتوى الحالي
                for i in range(self.general_chat_layout.count()):
                    item = self.general_chat_layout.itemAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # إعادة عرض الرسائل
                for msg in self.general_chat_history:
                    bubble_type = "user" if msg.get('role') == 'user' else "bot"
                    self.add_general_bubble(msg.get('content', ''), bubble_type)
                
                QMessageBox.information(self, "نجاح", "تم تحميل المحادثة بنجاح!")
            else:
                QMessageBox.warning(self, "تنبيه", "لا يمكن تحميل هذه المحادثة")
        except Exception as e:
            QMessageBox.critical(self, "خطأ", f"فشل تحميل المحادثة: {str(e)}")
    
    def delete_conversation_item(self, conversation):
        """حذف محادثة من السجل"""
        reply = QMessageBox.question(
            self,
            "تأكيد الحذف",
            f"هل أنت متأكد من حذف المحادثة:\n{conversation.get('title', 'بدون عنوان')}؟",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.history_manager.delete_conversation(conversation.get('id')):
                QMessageBox.information(self, "نجاح", "تم حذف المحادثة")
                # إعادة تحميل صفحة المحفوظات
                self.refresh_history_page()
            else:
                QMessageBox.critical(self, "خطأ", "فشل حذف المحادثة")
    
    def save_current_conversation(self, title="محادثة عامة"):
        """حفظ المحادثة الحالية إلى السجل"""
        try:
            conversation_data = {
                "title": title,
                "type": "general_chat",
                "messages": self.general_chat_history
            }
            self.history_manager.save_conversation(conversation_data)
            return True
        except Exception as e:
            print(f"خطأ في حفظ المحادثة: {e}")
            return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # قائمة الخطوط المفضلة مع احتياطيات
    preferred_fonts = ["itf Qomra Arabic", "Arial", "Tahoma", "Segoe UI", "MS Sans Serif"]
    
    selected_font = None
    font_db = QFontDatabase()
    
    # البحث عن أول خط متاح
    families = font_db.families()
    for font_name in preferred_fonts:
        if font_name in families:
            selected_font = font_name
            break
    
    # في حال عدم توفر أي من الخطوط المفضلة، استخدم الخط الافتراضي
    if selected_font is None:
        selected_font = "Arial"  # احتياطي نهائي
    
    font = QFont(selected_font, 18)
    app.setFont(font)
    
    window = ModernApp()
    window.show()
    sys.exit(app.exec())
