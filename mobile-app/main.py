import flet as ft
import requests
import base64
import json
from pathlib import Path

class DiamondGraderApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "AutoGemGrade"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 20
        self.page.scroll = "adaptive"
        
        # Edge device configuration
        self.edge_url = "http://RASPBERRY_PI_IP:8001"
        
        # UI Components
        self.upload_button = None
        self.image_display = None
        self.result_container = None
        self.loading_indicator = None
        self.selected_image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Title
        title = ft.Text(
            "AutoGemGrade",
            size=32,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.BLUE_700
        )
        
        subtitle = ft.Text(
            "AI-Powered Diamond Grading System",
            size=16,
            color=ft.colors.GREY_700
        )
        
        # File picker for image upload
        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)
        
        # Upload button
        self.upload_button = ft.ElevatedButton(
            "Upload Diamond Image",
            icon=ft.icons.UPLOAD_FILE,
            on_click=lambda _: self.file_picker.pick_files(
                allowed_extensions=["jpg", "jpeg", "png"],
                dialog_title="Select Diamond Image"
            ),
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE_700,
                padding=20,
            ),
            height=50,
        )
        
        # Instructions text
        instructions = ft.Container(
            content=ft.Column([
                ft.Text("Instructions:", weight=ft.FontWeight.BOLD, size=14),
                ft.Text("1. Click 'Upload Diamond Image' to select an image", size=12),
                ft.Text("2. Choose a clear image of a diamond", size=12),
                ft.Text("3. Wait for analysis (typically 5-15 seconds)", size=12),
                ft.Text("4. View the results showing cut, shape, and color", size=12),
            ]),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.BLUE_50,
            margin=ft.margin.only(bottom=20)
        )
        
        # Image display
        self.image_display = ft.Container(
            content=ft.Icon(ft.icons.IMAGE, size=100, color=ft.colors.GREY_400),
            width=400,
            height=400,
            border=ft.border.all(2, ft.colors.GREY_300),
            border_radius=10,
            alignment=ft.alignment.center,
            margin=ft.margin.only(top=20, bottom=20)
        )
        
        # Loading indicator
        self.loading_indicator = ft.Container(
            content=ft.Column([
                ft.ProgressRing(),
                ft.Text("Analyzing diamond...", size=14, color=ft.colors.GREY_700)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            visible=False,
            margin=ft.margin.only(top=10, bottom=10)
        )
        
        # Results container
        self.result_container = ft.Container(
            visible=False,
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.BLUE_50,
            margin=ft.margin.only(top=20)
        )
        
        # Analyze button (initially disabled)
        self.analyze_button = ft.ElevatedButton(
            "Analyze Diamond",
            icon=ft.icons.ANALYTICS,
            on_click=self.analyze_image,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.GREEN_700,
            ),
            height=50,
        )
        
        # Layout
        self.page.add(
            ft.Column([
                ft.Row([title], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([subtitle], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(height=20, color=ft.colors.TRANSPARENT),
                instructions,
                ft.Row([self.upload_button], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([self.image_display], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([self.analyze_button], alignment=ft.MainAxisAlignment.CENTER),
                self.loading_indicator,
                self.result_container,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )
    
    def on_file_selected(self, e: ft.FilePickerResultEvent):
        """Handle file selection"""
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            self.selected_image_path = file_path
            self.display_selected_image(file_path)
            self.analyze_button.disabled = False
            self.result_container.visible = False
            self.page.update()
    
    def display_selected_image(self, image_path):
        """Display the selected image"""
        try:
            # Read and encode image for display
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            # Display image
            self.image_display.content = ft.Image(
                src_base64=img_base64,
                width=400,
                height=400,
                fit=ft.ImageFit.CONTAIN
            )
            
            self.show_message(f"Image loaded: {Path(image_path).name}")
            self.page.update()
            
        except Exception as ex:
            self.show_message(f"Error loading image: {str(ex)}")
    
    def analyze_image(self, e):
        """Analyze the uploaded image"""
        if not self.selected_image_path:
            self.show_message("Please upload an image first")
            return
        
        try:
            # Read and encode image
            with open(self.selected_image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            # Show loading
            self.loading_indicator.visible = True
            self.result_container.visible = False
            self.analyze_button.disabled = True
            self.page.update()
            
            # Send to edge device
            self.send_to_edge(img_base64)
            
        except Exception as ex:
            self.show_message(f"Error processing image: {str(ex)}")
            self.loading_indicator.visible = False
            self.analyze_button.disabled = False
            self.page.update()
    
    def send_to_edge(self, img_base64):
        """Send image to Raspberry Pi edge device"""
        try:
            response = requests.post(
                f"{self.edge_url}/process",
                json={"image": img_base64},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                if results.get("success", False):
                    self.display_results(results)
                else:
                    self.show_message(f"Analysis failed: {results.get('error', 'Unknown error')}")
                    self.analyze_button.disabled = False
            else:
                self.show_message(f"Server error: {response.status_code}")
                self.analyze_button.disabled = False
                
        except requests.exceptions.ConnectionError:
            self.show_message("Cannot connect to edge device. Please check connection.")
            self.analyze_button.disabled = False
        except requests.exceptions.Timeout:
            self.show_message("Request timed out. Please try again.")
            self.analyze_button.disabled = False
        except requests.exceptions.RequestException as ex:
            self.show_message(f"Connection error: {str(ex)}")
            self.analyze_button.disabled = False
        finally:
            self.loading_indicator.visible = False
            self.page.update()
    
    def display_results(self, results):
        """Display the diamond grading results"""
        cut = results.get("cut", {})
        shape = results.get("shape", {})
        color = results.get("color", {})
        
        self.result_container.content = ft.Column([
            ft.Row([
                ft.Icon(ft.icons.CHECK_CIRCLE, color=ft.colors.GREEN, size=30),
                ft.Text("Diamond Analysis Results", size=20, weight=ft.FontWeight.BOLD),
            ]),
            ft.Divider(),
            self.create_result_row("Cut", cut.get("prediction", "N/A"), 
                                  cut.get("confidence", 0)),
            self.create_result_row("Shape", shape.get("prediction", "N/A"), 
                                  shape.get("confidence", 0)),
            self.create_result_row("Color", color.get("prediction", "N/A"), 
                                  color.get("confidence", 0)),
            ft.Divider(),
            ft.Text(
                f"Overall Confidence: {results.get('overall_confidence', 0):.1f}%",
                size=14,
                weight=ft.FontWeight.BOLD,
                color=ft.colors.BLUE_700
            )
        ])
        
        self.result_container.visible = True
        self.analyze_button.disabled = False
        self.page.update()
    
    def create_result_row(self, label, value, confidence):
        """Create a result display row"""
        # Determine confidence color
        if confidence >= 90:
            conf_color = ft.colors.GREEN
        elif confidence >= 75:
            conf_color = ft.colors.ORANGE
        else:
            conf_color = ft.colors.RED
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"{label}:", weight=ft.FontWeight.BOLD, size=16, width=100),
                    ft.Text(value, size=16, color=ft.colors.BLUE_700, weight=ft.FontWeight.W_500),
                ]),
                ft.Row([
                    ft.Text("Confidence:", size=14, width=100),
                    ft.ProgressBar(value=confidence/100, width=200, color=conf_color),
                    ft.Text(f"{confidence:.1f}%", size=14, color=conf_color, weight=ft.FontWeight.BOLD),
                ])
            ]),
            padding=10,
            margin=ft.margin.only(bottom=10)
        )
    
    def show_message(self, message):
        """Show a message to the user"""
        self.page.snack_bar = ft.SnackBar(
            ft.Text(message),
            bgcolor=ft.colors.BLUE_700
        )
        self.page.snack_bar.open = True
        self.page.update()

def main(page: ft.Page):
    app = DiamondGraderApp(page)

ft.app(target=main)