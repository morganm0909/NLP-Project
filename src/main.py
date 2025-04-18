from text_loader import load_text_from_file, load_text_from_pdf, load_text_from_url
from preprocess import clean_text
from summarizer import summarize_text
from keypoints import extract_key_points
import nltk
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit, QInputDialog, QComboBox, QLineEdit, QProgressBar, QGroupBox, QFormLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal

style_sheet = """
QMainWindow {
    background-color: #E4EDFA;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}

QLabel {
    font-size: 20px;
    color: #333333;
    margin-bottom: 4px;
}

QLineEdit, QComboBox, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 8px;
    font-size: 14px;
    color: #222;
}

QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #0078d4;
    background-color: #fcfcfc;
}

QPushButton {
    background-color: #FF99FA;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    border-radius: 8px;
    margin-top: 6px;
}

QPushButton:hover {
    background-color: #FF21FC;
}

QPushButton:pressed {
    background-color: #004f8a;
}

QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    text-align: center;
    background-color: #e9eef4;
    height: 20px;
    font-size: 12px;
    color: #333;
}

QProgressBar::chunk {
    background-color: #3AAD47;
    border-radius: 10px;
}

QComboBox {
    padding: 6px;
    border: 1px solid #ccc;
    border-radius: 5px;
   


QComboBox QAbstractItemView {
    background-color: #2f2f2f;
    color: white;
    selection-background-color: #FF99FA;  /* highlight color */
    selection-color: black;
    padding: 5px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #FFC8FF;  /* hover color */
    color: black;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 3px;
}

"""



class TextSummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text Summarizer')
        self.setGeometry(100, 100, 800, 600)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Title
        self.title = QLabel("📝 Text Summarizer Tool")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-bottom: 20px;")
        main_layout.addWidget(self.title, alignment=Qt.AlignCenter)

        # Input Section
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)

        input_label = QLabel("Select Input Source:")
        input_label.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(input_label)

        self.source_selector = QComboBox()
        self.source_selector.addItems(["File", "Pdf", "Url"])
        input_layout.addWidget(self.source_selector)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter file path, PDF path, or URL")
        input_layout.addWidget(self.input_field)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        input_layout.addWidget(self.browse_button)

        main_layout.addLayout(input_layout)

        # Raw Text Display
        text_input_group = QGroupBox("Raw Text Input")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Loaded or pasted raw text will appear here...")
        text_layout.addWidget(self.text_edit)
        text_input_group.setLayout(text_layout)
        main_layout.addWidget(text_input_group)

        # Analyze Button
        self.analyze_button = QPushButton("🔍 Analyze")
        self.analyze_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.analyze_button.clicked.connect(self.analyze_text)
        main_layout.addWidget(self.analyze_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Summary Output
        summary_group = QGroupBox("📄 Summary:")
        summary_layout = QVBoxLayout()
        # summary_label.setStyleSheet("font-weight: bold;")
        # main_layout.addWidget(summary_label)

        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("Summary will appear here...")
        summary_layout.addWidget(self.summary_output)
        summary_group.setLayout(summary_layout)
        main_layout.addWidget(summary_group)

        # Key Points Output
        keypoints_group = QGroupBox("🧠 Key Points:")
        # keypoints_label.setStyleSheet("font-weight: bold;")
        keypoints_layout = QVBoxLayout()
        # main_layout.addWidget(keypoints_label)

        self.keypoints_output = QTextEdit()
        self.keypoints_output.setReadOnly(True)
        self.keypoints_output.setPlaceholderText("Extracted key points will appear here...")
        keypoints_layout.addWidget(self.keypoints_output)
        keypoints_group.setLayout(keypoints_layout)
        main_layout.addWidget(keypoints_group)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


    def browse_file(self):
        source = self.source_selector.currentText()
        if source in ["File", "Pdf"]:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
            if file_path:
                self.input_field.setText(file_path)
            
        else:
            QMessageBox.information(self, "Browse", "URL must be entered manually.")
            
    
        if source == "File":
            text = load_text_from_file(file_path)
            self.text_edit.setPlainText(text)
        elif source == "Pdf":
            text = load_text_from_pdf(file_path)
            self.text_edit.setPlainText(text)
        elif source == "Url":
            text = load_text_from_url(file_path)
            self.text_edit.setPlainText(text)
        else:
            raise ValueError("Invalid source type")


    def analyze_text(self):
        source = self.source_selector.currentText()
        path_or_url = self.input_field.text().strip()
        text = self.text_edit.toPlainText()
        
        self.progress_bar.setValue(0)

        self.worker = AnalysisWorker(source, path_or_url)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()

            

        # cleaned = clean_text(text)
        # summary = summarize_text(cleaned)
        # key_points = extract_key_points(cleaned)

        # self.summary_output.setPlainText(summary)
        # self.keypoints_output.setPlainText('\n'.join(key_points))

            
    def display_results(self, summary, key_points):
        self.summary_output.setPlainText(summary)
        self.keypoints_output.setPlainText('\n'.join(key_points))

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setValue(0)

    # def summarize_text(self):
    #     input_text = self.text_edit.toPlainText()
        
    #     if not input_text.strip():
    #         QMessageBox.warning(self, 'Warning', 'No text loaded for summarization.')
    #         return

    #     cleaned_text = clean_text(input_text)
        
    #     summary = summarize_text(cleaned_text)
        
    #     key_points = extract_key_points(cleaned_text)
        
    #     self.summary_output.setPlainText(summary)
    #     self.keypoints_output.setPlainText('\n'.join(key_points))
class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, source, path_or_url):
        super().__init__()
        self.source = source
        self.path_or_url = path_or_url

    def run(self):
        try:
            self.progress.emit(10)
            if self.source == "File":
                text = load_text_from_file(self.path_or_url)
            elif self.source == "Pdf":
                text = load_text_from_pdf(self.path_or_url)
            elif self.source == "Url":
                text = load_text_from_url(self.path_or_url)
            else:
                raise ValueError("Invalid source type")

            self.progress.emit(40)
            cleaned = clean_text(text)

            self.progress.emit(70)
            summary = summarize_text(cleaned)

            self.progress.emit(85)
            key_points = extract_key_points(cleaned)

            self.progress.emit(100)
            self.finished.emit(summary, key_points)

        except Exception as e:
            self.error.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheet)

    window = TextSummarizerApp()
    window.show()
    sys.exit(app.exec_())
