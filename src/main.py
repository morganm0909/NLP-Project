from text_loader import load_text_from_file, load_text_from_pdf, load_text_from_url
from preprocess import clean_text
from summarizer import summarize_text
from keypoints import extract_key_points
import nltk
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit, QInputDialog, QComboBox, QLineEdit, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class TextSummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text Summarizer')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel('Load text from:')
        layout.addWidget(self.label)
        
        self.source_selector = QComboBox()
        self.source_selector.addItems(["File", "Pdf", "Url"])
        layout.addWidget(QLabel("Select Input Source:"))
        layout.addWidget(self.source_selector)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter file path, PDF path, or URL")
        layout.addWidget(self.input_field)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

        # self.file_button = QPushButton('File')
        # self.file_button.clicked.connect(self.load_file)
        # layout.addWidget(self.file_button)

        # self.pdf_button = QPushButton('PDF')
        # self.pdf_button.clicked.connect(self.load_pdf)
        # layout.addWidget(self.pdf_button)

        # self.url_button = QPushButton('URL')
        # self.url_button.clicked.connect(self.load_url)
        # layout.addWidget(self.url_button)

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        # self.summarize_button = QPushButton('Summarize')
        # self.summarize_button.clicked.connect(self.summarize_text)
        # layout.addWidget(self.summarize_button)
        
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze_text)
        layout.addWidget(analyze_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        layout.addWidget(QLabel("Summary:"))
        layout.addWidget(self.summary_output)
        
        self.keypoints_output = QTextEdit()
        self.keypoints_output.setReadOnly(True)
        layout.addWidget(QLabel("Key Points:"))
        layout.addWidget(self.keypoints_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # def load_file(self):
    #     options = QFileDialog.Options()
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt);;All Files (*)", options=options)
    #     if file_path:
    #         text = load_text_from_file(file_path)
    #         self.text_edit.setPlainText(text)

    # def load_pdf(self):
    #     options = QFileDialog.Options()
    #     pdf_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf);;All Files (*)", options=options)
    #     if pdf_path:
    #         text = load_text_from_pdf(pdf_path)
    #         self.text_edit.setPlainText(text)

    # def load_url(self):
    #     url, ok = QInputDialog.getText(self, 'Input URL', 'Enter URL:')
    #     if ok and url:
    #         text = load_text_from_url(url)
    #         self.text_edit.setPlainText(text)
    
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
    window = TextSummarizerApp()
    window.show()
    sys.exit(app.exec_())
