#!/usr/bin/env python3
"""
Generates a PDF report summarizing the video analysis data stored in the SQLite database.
"""

import sqlite3
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage # Avoid name clash with reportlab Image
import io
import argparse
import platform # For basic system info
import psutil   # For RAM info
import cpuinfo  # For CPU info

# Optional GPU detection
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil library not found. GPU information will not be included.")

def bytes_to_gb(bytes_val):
    """Converts bytes to gigabytes."""
    return bytes_val / (1024**3)

def get_hardware_info():
    """Gathers CPU, RAM, and GPU information."""
    info = {
        'cpu': 'N/A',
        'ram_total_gb': 'N/A',
        'gpus': []
    }
    try:
        # CPU Info
        try:
            # Prefer py-cpuinfo for detailed name
            info['cpu'] = cpuinfo.get_cpu_info()['brand_raw']
        except Exception as e_cpu:
            print(f"Could not get detailed CPU info via cpuinfo: {e_cpu}. Falling back to platform.")
            # Fallback using platform
            info['cpu'] = platform.processor()
        
        # RAM Info
        try:
            mem = psutil.virtual_memory()
            info['ram_total_gb'] = f"{bytes_to_gb(mem.total):.2f} GB"
        except Exception as e_ram:
             print(f"Could not get RAM info via psutil: {e_ram}.")

        # GPU Info (Optional)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for gpu in gpus:
                        info['gpus'].append({
                            'name': gpu.name,
                            'memory_gb': f"{bytes_to_gb(gpu.memoryTotal * 1024 * 1024):.2f} GB" # Convert MB to GB
                        })
                else:
                    info['gpus'].append({'name': 'No compatible GPUs detected by GPUtil', 'memory_gb': 'N/A'})
            except Exception as e_gpu:
                print(f"Error getting GPU info via GPUtil: {e_gpu}")
                info['gpus'].append({'name': f'Error detecting GPU ({e_gpu})', 'memory_gb': 'N/A'})
        else:
             info['gpus'].append({'name': 'GPUtil not installed', 'memory_gb': 'N/A'})

    except Exception as e_hw:
        print(f"An unexpected error occurred while fetching hardware info: {e_hw}")
        # Ensure default values remain if a major error occurs
        info['cpu'] = info.get('cpu', 'Error fetching')
        info['ram_total_gb'] = info.get('ram_total_gb', 'Error fetching')
        if not info['gpus']:
             info['gpus'].append({'name': 'Error fetching', 'memory_gb': 'N/A'})
             
    return info

def resize_image(image_path, max_width=4*inch, max_height=4*inch):
    """Resizes an image to fit within max dimensions while preserving aspect ratio."""
    try:
        img = PILImage.open(image_path)
        img.thumbnail((max_width, max_height))
        
        # Convert image to RGB if it has an alpha channel (RGBA) for ReportLab compatibility
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Save resized image to a byte stream
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG') # Use JPEG for compatibility
        img_byte_arr.seek(0)
        
        # Return the BytesIO stream directly along with dimensions
        return img_byte_arr, img.width, img.height
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, 0, 0

def create_report(db_path, report_path="video_analysis_report.pdf", output_dir="processed_data"):
    """
    Generates the PDF report.
    
    Args:
        db_path: Path to the SQLite database file.
        report_path: Path to save the generated PDF report.
        output_dir: Base directory where processed data (frames, audio) is stored.
    """
    conn = None
    try:
        # --- Get Hardware Info --- 
        hardware_info = get_hardware_info()
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch all videos - ADD granular timings
        cursor.execute("""SELECT id, file_path, duration, 
                           processing_time_seconds, 
                           frame_processing_time_seconds, 
                           audio_processing_time_seconds 
                       FROM videos ORDER BY id""")
        videos = cursor.fetchall()

        if not videos:
            print("No videos found in the database.")
            return

        # Setup PDF document
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # --- Add Report Title --- 
        story.append(Paragraph("Video Analysis Report", styles['h1']))
        story.append(Spacer(1, 0.2*inch))

        # --- Add Hardware Info Section --- 
        story.append(Paragraph("Hardware Information", styles['h2']))
        story.append(Paragraph(f"CPU: {hardware_info['cpu']}", styles['Normal']))
        story.append(Paragraph(f"RAM: {hardware_info['ram_total_gb']}", styles['Normal']))
        story.append(Paragraph("GPU(s):", styles['Normal']))
        if hardware_info['gpus']:
            for gpu in hardware_info['gpus']:
                story.append(Paragraph(f"- {gpu['name']} ({gpu['memory_gb']})", styles['Normal']))
        else:
             story.append(Paragraph("- N/A", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # --- Loop Through Videos --- 
        for video in videos:
            video_id = video['id']
            video_path = video['file_path']
            video_duration = video['duration']
            total_processing_time = video['processing_time_seconds'] 
            frame_processing_time = video['frame_processing_time_seconds']
            audio_processing_time = video['audio_processing_time_seconds']
            
            story.append(Paragraph(f"Video {video_id}: {os.path.basename(video_path)}", styles['h2']))
            story.append(Paragraph(f"Source Path: {video_path}", styles['Normal']))
            story.append(Paragraph(f"Duration: {video_duration:.2f} seconds", styles['Normal']))
            
            # Display processing times
            if total_processing_time is not None:
                 story.append(Paragraph(f"Total Processing Time: {total_processing_time:.2f} seconds", styles['Normal']))
                 # Display granular times if available and total time is known
                 if frame_processing_time is not None:
                      story.append(Paragraph(f"  - Frames/Description Time: {frame_processing_time:.2f}s", styles['Normal']))
                 if audio_processing_time is not None:
                      story.append(Paragraph(f"  - Audio/Transcription Time: {audio_processing_time:.2f}s", styles['Normal']))
                 # You could add a check here: if total_processing_time != frame_time + audio_time, report discrepancy or "other time"
            else:
                 story.append(Paragraph("Processing Time: Not recorded", styles['Italic']))
                 
            story.append(Spacer(1, 0.1*inch))

            # Fetch key frames for this video
            cursor.execute("""
                SELECT id, timestamp, frame_path, description 
                FROM key_frames 
                WHERE video_id = ? 
                ORDER BY timestamp
            """, (video_id,))
            frames = cursor.fetchall()

            story.append(Paragraph("Key Frames:", styles['h3']))
            if frames:
                for frame in frames:
                    frame_path = frame['frame_path']
                    # Make frame path relative to output_dir if it's absolute
                    if os.path.isabs(frame_path):
                        try:
                           frame_path = os.path.relpath(frame_path, os.path.dirname(output_dir)) # Adjust if needed
                        except ValueError:
                            # Handle cases where paths are on different drives (Windows)
                            pass 
                    frame_path = "./" + frame_path.replace("\\", "/")
                    print(f"frame_path: {frame_path}")        
                    full_frame_path = frame_path#os.path.join(os.path.dirname(db_path), frame_path) # Construct full path relative to db location
                    
                    story.append(Paragraph(f"- Frame ID: {frame['id']} at {frame['timestamp']:.2f}s", styles['Normal']))
                    story.append(Paragraph(f"  Description: {frame['description']}", styles['Normal']))
                    story.append(Paragraph(f"  Path: {frame_path}", styles['Code']))
                    
                    # Try adding the image thumbnail
                    if os.path.exists(full_frame_path):
                        print("exists")
                        # resize_image now returns a BytesIO stream
                        resized_img_stream, width, height = resize_image(full_frame_path)
                        if resized_img_stream: # Check if resize_image succeeded
                            # Pass the BytesIO stream directly to reportlab.platypus.Image
                            img = Image(resized_img_stream, width=width, height=height)
                            img.hAlign = 'LEFT'
                            story.append(img)
                            story.append(Spacer(1, 0.1*inch))
                    else:
                        print("not exists")
                        story.append(Paragraph(f"  (Image not found at: {full_frame_path})", styles['Italic']))
                    
                    story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph("No key frames found for this video.", styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))

            # Fetch audio transcriptions for this video
            cursor.execute("""
                SELECT id, start_time, end_time, text 
                FROM audio_transcriptions 
                WHERE video_id = ? 
                ORDER BY start_time
            """, (video_id,))
            transcriptions = cursor.fetchall()

            # Construct expected audio file path
            audio_filename = f"video_{video_id}_audio.wav"
            # Make audio path relative if necessary (similar logic to frames)
            audio_rel_path = os.path.join("audio", audio_filename) 
            full_audio_path = os.path.join(os.path.dirname(db_path), audio_rel_path)

            story.append(Paragraph("Audio Transcription:", styles['h3']))
            story.append(Paragraph(f"Expected Audio File: {audio_rel_path}", styles['Code']))
            if os.path.exists(full_audio_path):
                 story.append(Paragraph("(Audio file exists)", styles['Italic']))
            else:
                 story.append(Paragraph(f"(Audio file not found at: {full_audio_path})", styles['Italic']))


            if transcriptions:
                for trans in transcriptions:
                    story.append(Paragraph(f"- ID {trans['id']} ({trans['start_time']:.2f}s - {trans['end_time']:.2f}s): {trans['text']}", styles['Normal']))
            else:
                story.append(Paragraph("No audio transcriptions found for this video.", styles['Normal']))

            story.append(PageBreak()) # Add a page break between videos

        # Remove the last page break if it exists
        if story and isinstance(story[-1], PageBreak):
            story.pop()

        # Build the PDF
        doc.build(story)
        print(f"Report successfully generated: {report_path}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a PDF report from video analysis data.")
    parser.add_argument("--output-dir", default="processed_data", help="Directory containing the processed data (database, frames, audio). Default: processed_data")
    parser.add_argument("--db-name", default="video_search.db", help="Name of the SQLite database file within the output directory. Default: video_search.db")
    parser.add_argument("--report-name", default="video_analysis_report.pdf", help="Name of the output PDF report file. Default: video_analysis_report.pdf")
    
    args = parser.parse_args()

    db_full_path = os.path.join(args.output_dir, args.db_name)
    report_full_path = os.path.join(args.output_dir, args.report_name) # Save report in output dir

    if not os.path.exists(db_full_path):
        print(f"Error: Database file not found at {db_full_path}")
    else:
        create_report(db_full_path, report_full_path, args.output_dir) 