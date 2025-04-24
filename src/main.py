from text_loader import load_text_from_file, load_text_from_pdf, load_text_from_url
from preprocess import clean_text
from summarizer import summarize_text, SummarizerFactory, evaluate_summary
from keypoints import extract_key_points
import nltk
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox, 
                            QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QLabel, 
                            QPushButton, QTextEdit, QInputDialog, QComboBox, 
                            QLineEdit, QProgressBar, QSlider, QGroupBox, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QDialog, QTableWidget, 
                            QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import transformers  # Make sure to import this for the callback class

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



class FineTuningDialog(QDialog):
    """Dialog for fine-tuning a summarization model on custom data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Fine-Tune Summarization Model")
        self.setMinimumWidth(600)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Base model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Base Model:"))
        self.base_model_selector = QComboBox()
        self.base_model_selector.addItems(["facebook/bart-large-cnn", "t5-base", "google/pegasus-xsum"])
        model_layout.addWidget(self.base_model_selector)
        layout.addLayout(model_layout)
        
        # Training data
        train_group = QGroupBox("Training Data")
        train_layout = QGridLayout()
        
        # Text-summary pairs
        self.pair_table = QTableWidget(0, 2)
        self.pair_table.setHorizontalHeaderLabels(["Original Text", "Summary"])
        self.pair_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        train_layout.addWidget(self.pair_table, 0, 0, 1, 3)
        
        # Add/remove pair buttons
        add_pair_btn = QPushButton("Add Pair")
        add_pair_btn.clicked.connect(self.add_empty_pair)
        train_layout.addWidget(add_pair_btn, 1, 0)
        
        remove_pair_btn = QPushButton("Remove Selected")
        remove_pair_btn.clicked.connect(self.remove_selected_pair)
        train_layout.addWidget(remove_pair_btn, 1, 1)
        
        # Import from file
        import_btn = QPushButton("Import from CSV")
        import_btn.clicked.connect(self.import_from_csv)
        train_layout.addWidget(import_btn, 1, 2)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Training parameters
        param_group = QGroupBox("Training Parameters")
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10)
        self.epochs_spin.setValue(3)
        param_layout.addWidget(self.epochs_spin, 0, 1)
        
        param_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(4)
        param_layout.addWidget(self.batch_spin, 1, 1)
        
        param_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        out_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit("./fine_tuned_model")
        out_dir_btn = QPushButton("Browse")
        out_dir_btn.clicked.connect(self.browse_output_dir)
        out_dir_layout.addWidget(self.output_dir)
        out_dir_layout.addWidget(out_dir_btn)
        param_layout.addLayout(out_dir_layout, 2, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Progress indicators
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Fine-Tuning")
        self.train_button.clicked.connect(self.start_fine_tuning)
        buttons_layout.addWidget(self.train_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # Add initial empty rows
        for _ in range(3):
            self.add_empty_pair()
            
    def add_empty_pair(self):
        """Add an empty row to the table"""
        row = self.pair_table.rowCount()
        self.pair_table.insertRow(row)
        
        # Add text cells
        text_cell = QTextEdit()
        summary_cell = QTextEdit()
        
        self.pair_table.setCellWidget(row, 0, text_cell)
        self.pair_table.setCellWidget(row, 1, summary_cell)
        
    def remove_selected_pair(self):
        """Remove selected row from the table"""
        selected_rows = self.pair_table.selectedIndexes()
        if not selected_rows:
            return
            
        # Get unique row indices and sort in descending order to avoid index shifting
        rows = set()
        for index in selected_rows:
            rows.add(index.row())
        
        for row in sorted(rows, reverse=True):
            self.pair_table.removeRow(row)
    
    def import_from_csv(self):
        """Import text-summary pairs from CSV file"""
        import csv
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "", "CSV Files (*.csv);;All Files (*)")
            
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Skip header row
                
                # Clear existing rows
                while self.pair_table.rowCount() > 0:
                    self.pair_table.removeRow(0)
                
                # Add rows from CSV
                for row_data in reader:
                    if len(row_data) >= 2:
                        row = self.pair_table.rowCount()
                        self.pair_table.insertRow(row)
                        
                        text_cell = QTextEdit()
                        text_cell.setPlainText(row_data[0])
                        
                        summary_cell = QTextEdit()
                        summary_cell.setPlainText(row_data[1])
                        
                        self.pair_table.setCellWidget(row, 0, text_cell)
                        self.pair_table.setCellWidget(row, 1, summary_cell)
            
            QMessageBox.information(self, "Import Successful", 
                                   f"Imported {self.pair_table.rowCount()} text-summary pairs.")
                                   
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing CSV: {str(e)}")
    
    def browse_output_dir(self):
        """Select output directory for fine-tuned model"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def get_training_data(self):
        """Extract text-summary pairs from the table"""
        texts = []
        summaries = []
        
        for row in range(self.pair_table.rowCount()):
            text_widget = self.pair_table.cellWidget(row, 0)
            summary_widget = self.pair_table.cellWidget(row, 1)
            
            if text_widget and summary_widget:
                text = text_widget.toPlainText().strip()
                summary = summary_widget.toPlainText().strip()
                
                if text and summary:
                    texts.append(text)
                    summaries.append(summary)
        
        return texts, summaries
    
    def start_fine_tuning(self):
        """Start the fine-tuning process"""
        # Get training data
        texts, summaries = self.get_training_data()
        
        if len(texts) < 3:
            QMessageBox.warning(self, "Insufficient Data", 
                               "Please provide at least 3 text-summary pairs for fine-tuning.")
            return
        
        # Get parameters
        base_model = self.base_model_selector.currentText()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        output_dir = self.output_dir.text()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Confirm with user
        msg = f"Start fine-tuning with:\n" \
              f"- Base model: {base_model}\n" \
              f"- Training examples: {len(texts)}\n" \
              f"- Epochs: {epochs}\n" \
              f"- Output directory: {output_dir}\n\n" \
              f"This process may take a long time depending on your hardware. Continue?"
              
        reply = QMessageBox.question(self, "Confirm Fine-Tuning", msg, 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                    
        if reply == QMessageBox.No:
            return
            
        # Disable UI during training
        self.setEnabled(False)
        self.train_button.setText("Training...")
        self.progress_bar.setValue(10)
        self.status_label.setText("Initializing fine-tuning...")
        
        # Run fine-tuning in a separate thread
        self.train_thread = FineTuningWorker(
            base_model, texts, summaries, output_dir, epochs, batch_size)
        self.train_thread.progress.connect(self.progress_bar.setValue)
        self.train_thread.status.connect(self.status_label.setText)
        self.train_thread.finished.connect(self.training_finished)
        self.train_thread.error.connect(self.training_error)
        self.train_thread.start()
    
    def training_finished(self, model_path):
        """Handle successful training completion"""
        self.setEnabled(True)
        self.train_button.setText("Start Fine-Tuning")
        self.progress_bar.setValue(100)
        self.status_label.setText("Fine-tuning completed!")
        
        # Ask if user wants to use this model now
        msg = f"Fine-tuning completed successfully!\n\n" \
              f"Model saved to: {model_path}\n\n" \
              f"Would you like to use this model now?"
              
        reply = QMessageBox.question(self, "Fine-Tuning Complete", msg, 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                                    
        if reply == QMessageBox.Yes and self.parent:
            # Set model in main window
            self.parent.model_selector.setCurrentText("custom")
            self.parent.custom_model_path.setText(model_path)
            
        self.accept()
    
    def training_error(self, error_msg):
        """Handle training error"""
        self.setEnabled(True)
        self.train_button.setText("Start Fine-Tuning")
        self.progress_bar.setValue(0)
        self.status_label.setText("Fine-tuning failed!")
        
        QMessageBox.critical(self, "Fine-Tuning Error", 
                            f"An error occurred during fine-tuning:\n{error_msg}")


class FineTuningWorker(QThread):
    """Worker thread for fine-tuning models"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, base_model, texts, summaries, output_dir, epochs, batch_size):
        super().__init__()
        self.base_model = base_model
        self.texts = texts
        self.summaries = summaries
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        
    def run(self):
        try:
            from summarizer import FineTunedSummarizer
            
            self.progress.emit(15)
            self.status.emit("Initializing fine-tuning...")
            
            # Create and initialize summarizer
            summarizer = FineTunedSummarizer(self.base_model)
            
            self.progress.emit(20)
            self.status.emit("Preparing training data...")
            
            # Custom progress callback
            def progress_callback(step, total_steps):
                progress = 20 + int(70 * (step / total_steps))
                self.progress.emit(progress)
                self.status.emit(f"Training: Step {step}/{total_steps}")
            
            # Fine-tune the model
            self.status.emit("Starting fine-tuning...")
            summarizer.fine_tune(
                train_texts=self.texts,
                train_summaries=self.summaries,
                output_dir=self.output_dir,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callback=progress_callback
            )
            
            self.progress.emit(95)
            self.status.emit("Saving fine-tuned model...")
            
            self.finished.emit(self.output_dir)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class AnalysisWorker(QThread):
    """Worker thread for text analysis"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, list, dict)
    error = pyqtSignal(str)

    def __init__(self, text, model_type, custom_path=None, ratio=0.3):
        super().__init__()
        self.text = text
        self.model_type = model_type
        self.custom_path = custom_path
        self.ratio = ratio

    def run(self):
        try:
            self.progress.emit(10)
            
            # Clean text
            cleaned = clean_text(self.text)
            if not cleaned or len(cleaned.split()) < 20:
                raise ValueError("Insufficient content for summarization.")
            self.progress.emit(20)
            print("CLEANED CONTENT:\n", cleaned[:500])  # Just the first 500 chars

            # Create summarizer based on selected model
            if self.model_type == "custom" and self.custom_path:
                summarizer = SummarizerFactory.get_summarizer("custom", self.custom_path)
            else:
                summarizer = SummarizerFactory.get_summarizer(self.model_type)
            
            self.progress.emit(30)
            
            # Generate summary
            summary = summarizer.summarize(cleaned, ratio=self.ratio)
            self.progress.emit(60)
            
            # Extract key points
            key_points = extract_key_points(cleaned)
            self.progress.emit(80)
            
            # Evaluate summary
            evaluation = evaluate_summary(cleaned, summary)
            self.progress.emit(100)
            
            self.finished.emit(summary, key_points, evaluation)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class TextSummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced Text Summarizer')
        self.setGeometry(100, 100, 1000, 800)

        main_layout = QVBoxLayout()
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        source_layout = QHBoxLayout()
        self.source_selector = QComboBox()
        self.source_selector.addItems(["File", "Pdf", "Url"])
        source_layout.addWidget(QLabel("Select Input Source:"))
        source_layout.addWidget(self.source_selector)
        input_layout.addLayout(source_layout)
        
        path_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter file path, PDF path, or URL")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        path_layout.addWidget(self.input_field)
        path_layout.addWidget(browse_button)
        input_layout.addLayout(path_layout)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Or paste text directly here...")
        input_layout.addWidget(self.text_edit)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Model configuration section
        model_group = QGroupBox("Summarization Model")
        model_layout = QGridLayout()
        
        # Model type selector
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_selector = QComboBox()
        self.model_selector.addItems(["bart", "t5", "pegasus", "led", "distilbart", "custom"])
        self.model_selector.currentTextChanged.connect(self.toggle_custom_model_input)
        model_layout.addWidget(self.model_selector, 0, 1)
        
        # Custom model path
        model_layout.addWidget(QLabel("Custom Model Path:"), 1, 0)
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setEnabled(False)
        model_layout.addWidget(self.custom_model_path, 1, 1)
        self.custom_model_browse = QPushButton("Browse")
        self.custom_model_browse.clicked.connect(self.browse_model)
        self.custom_model_browse.setEnabled(False)
        model_layout.addWidget(self.custom_model_browse, 1, 2)
        
        # Add fine-tuning button
        self.fine_tune_button = QPushButton("Fine-Tune New Model")
        self.fine_tune_button.clicked.connect(self.open_fine_tuning_dialog)
        model_layout.addWidget(self.fine_tune_button, 2, 0)
        
        # Summary length controls
        model_layout.addWidget(QLabel("Summary Length:"), 2, 1)
        length_layout = QHBoxLayout()
        self.length_slider = QSlider(Qt.Horizontal)
        self.length_slider.setRange(10, 50)
        self.length_slider.setValue(30)
        self.length_slider.setTickPosition(QSlider.TicksBelow)
        self.length_slider.setTickInterval(5)
        self.length_label = QLabel("30%")
        self.length_slider.valueChanged.connect(
            lambda v: self.length_label.setText(f"{v}%"))
        length_layout.addWidget(self.length_slider)
        length_layout.addWidget(self.length_label)
        model_layout.addLayout(length_layout, 2, 2)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Analysis button and progress bar
        analyze_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze and Summarize")
        self.analyze_button.clicked.connect(self.analyze_text)
        self.analyze_button.setMinimumHeight(40)
        analyze_layout.addWidget(self.analyze_button)
        main_layout.addLayout(analyze_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Results section
        results_layout = QHBoxLayout()
        
        # Summary output
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        summary_layout.addWidget(self.summary_output)
        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group)
        
        # Key points output
        keypoints_group = QGroupBox("Key Points")
        keypoints_layout = QVBoxLayout()
        self.keypoints_output = QTextEdit()
        self.keypoints_output.setReadOnly(True)
        keypoints_layout.addWidget(self.keypoints_output)
        keypoints_group.setLayout(keypoints_layout)
        results_layout.addWidget(keypoints_group)
        
        main_layout.addLayout(results_layout)
        
        # Evaluation section
        eval_group = QGroupBox("Summary Evaluation")
        eval_layout = QGridLayout()
        self.eval_output = QTextEdit()
        self.eval_output.setReadOnly(True)
        self.eval_output.setMaximumHeight(100)
        eval_layout.addWidget(self.eval_output, 0, 0)
        eval_group.setLayout(eval_layout)
        main_layout.addWidget(eval_group)
        
        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def toggle_custom_model_input(self, model_type):
        """Enable or disable custom model path input based on selection"""
        is_custom = model_type == "custom"
        self.custom_model_path.setEnabled(is_custom)
        self.custom_model_browse.setEnabled(is_custom)
        
    def open_fine_tuning_dialog(self):
        """Open the fine-tuning dialog"""
        dialog = FineTuningDialog(self)
        dialog.exec_()

    def browse_file(self):
        """Browse for input file"""
        source = self.source_selector.currentText()
        if source == "File":
            file_filter = "Text Files (*.txt);;All Files (*)"
        elif source == "Pdf":
            file_filter = "PDF Files (*.pdf);;All Files (*)"
        else:
            QMessageBox.information(self, "Browse", "URL must be entered manually.")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if file_path:
            self.input_field.setText(file_path)
            try:
                if source == "File":
                    text = load_text_from_file(file_path)
                    self.text_edit.setPlainText(text)
                elif source == "Pdf":
                    text = load_text_from_pdf(file_path)
                    self.text_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def browse_model(self):
        """Browse for custom model directory"""
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if model_dir:
            self.custom_model_path.setText(model_dir)

    def analyze_text(self):
        """Process the text and generate summary and key points"""
        # Get input text
        text = self.text_edit.toPlainText().strip()
        
        if not text:
            source = self.source_selector.currentText()
            path_or_url = self.input_field.text().strip()
            
            if not path_or_url:
                QMessageBox.warning(self, "Warning", "No text or input source provided.")
                return
                
            try:
                if source == "File":
                    text = load_text_from_file(path_or_url)
                elif source == "Pdf":
                    text = load_text_from_pdf(path_or_url)
                elif source == "Url":
                    text = load_text_from_url(path_or_url)
                else:
                    raise ValueError("Invalid source type")
                    
                self.text_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading content: {str(e)}")
                return
        
        # Get model configuration
        model_type = self.model_selector.currentText()
        custom_path = self.custom_model_path.text() if model_type == "custom" else None
        ratio = self.length_slider.value() / 100.0
        
        # Reset displays
        self.summary_output.clear()
        self.keypoints_output.clear()
        self.eval_output.clear()
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = AnalysisWorker(text, model_type, custom_path, ratio)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()
            
    def display_results(self, summary, key_points, evaluation):
        """Display analysis results"""
        self.summary_output.setPlainText(summary)
        self.keypoints_output.setPlainText('\n'.join(key_points))
        
        # Display evaluation metrics
        eval_text = "Summary Evaluation:\n"
        for metric, value in evaluation.items():
            eval_text += f"- {metric}: {value:.4f}\n"
        self.eval_output.setPlainText(eval_text)

    def show_error(self, message):
        """Display error message"""
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheet)

    window = TextSummarizerApp()
    window.show()
    sys.exit(app.exec_())