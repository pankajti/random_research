# create_video.py
import os
import re
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip, AudioFileClip, concatenate_videoclips,concatenate_audioclips
import numpy as np # For gTTS audio duration

# --- Configuration ---
TRANSCRIPT_FILE = 'explanation_script.txt'
OUTPUT_VIDEO_FILE = 'pytorch_nn_explanation.mp4'
TEMP_DIR = 'temp_video_assets'

# Video Settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 24

# Code Display Settings
FONT_PATH = "DejaVuSansMono.ttf" # Or 'DejaVuSansMono.ttf', 'FiraCode-Regular.ttf' - ensure this font exists on your system!
FONT_SIZE = 24
LINE_HEIGHT = FONT_SIZE + 4 # Padding between lines
CODE_X_OFFSET = 20
CODE_Y_OFFSET = 20
BACKGROUND_COLOR = (30, 30, 255)  # Dark gray
TEXT_COLOR = (240, 240, 240)  # Light gray
HIGHLIGHT_COLOR = (60, 60, 90, 180) # Darker blue with transparency for highlighting

# --- Ensure Temp Directory Exists ---
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Helper Functions ---

def get_audio_duration(file_path):
    """Get the duration of an audio file using moviepy's AudioFileClip."""
    try:
        audio_clip = AudioFileClip(file_path)
        duration = audio_clip.duration
        audio_clip.close() # Important to close the clip
        return duration
    except Exception as e:
        print(f"Error getting audio duration for {file_path}: {e}")
        return 2 # Default to 2 seconds if error

def create_code_image(code_lines, display_start_line, display_end_line, highlight_lines=None, image_index=0):
    """
    Generates an image of the code, optionally highlighting specific lines.
    `code_lines` is a list of all lines in the code file.
    `display_start_line` and `display_end_line` define the range to show.
    `highlight_lines` is a list of 0-indexed lines to highlight within the *full* code.
    """
    if highlight_lines is None:
        highlight_lines = []

    img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"Warning: Font not found at {FONT_PATH}. Using default font.")
        font = ImageFont.load_default()

    # Draw code lines
    current_y = CODE_Y_OFFSET
    for i, line_text in enumerate(code_lines):
        if display_start_line <= i <= display_end_line:
            # Draw highlight rectangle if this line is in highlight_lines
            if i in highlight_lines:
                # Calculate text width for this line to make highlight box snug
                text_bbox = draw.textbbox((0,0), line_text, font=font)
                line_width = text_bbox[2] - text_bbox[0] # Approximate width

                # Create a transparent overlay for highlighting
                overlay = Image.new('RGBA', img.size, (0,0,0,0))
                overlay_draw = ImageDraw.Draw(overlay)
                # Highlight rectangle (x1, y1, x2, y2)
                overlay_draw.rectangle(
                    [CODE_X_OFFSET - 5, current_y - 2, # Start slightly before text
                     CODE_X_OFFSET + line_width + 5, current_y + LINE_HEIGHT - 2], # End slightly after text
                    fill=HIGHLIGHT_COLOR
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img) # Re-draw on the new image

            draw.text((CODE_X_OFFSET, current_y), line_text, font=font, fill=TEXT_COLOR)
            current_y += LINE_HEIGHT

    img_path = os.path.join(TEMP_DIR, f'code_frame_{image_index:04d}.png')
    img.save(img_path)
    return img_path

def parse_transcript(transcript_file_path):
    """Parses the structured transcript file."""
    sections = []
    current_code_file = None

    try:
        with open(transcript_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Get code file path from the first line
        if lines and lines[0].strip().startswith('# CODE_FILE:'):
            current_code_file = lines[0].strip().split(':', 1)[1].strip()
            lines = lines[1:] # Remove the header line
        else:
            raise ValueError("Transcript must start with '# CODE_FILE: <path_to_code.py>'")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(r'\[(LINE|BLANK|ALL)\s*(.*?)?\]\s*(.*)', line, re.IGNORECASE)
            if match:
                marker_type = match.group(1).upper()
                marker_args = match.group(2).strip()
                text = match.group(3).strip()

                display_range = None
                highlight_lines = []

                if marker_type == 'LINE' or marker_type == 'ALL':
                    if marker_type == 'ALL':
                        display_range = 'ALL'
                    elif marker_args:
                        parts = marker_args.split(',')
                        range_str = parts[0].strip()
                        highlight_part = None
                        if len(parts) > 1 and 'highlight=' in parts[1]:
                             highlight_part = parts[1].strip()

                        if '-' in range_str:
                            start, end = map(int, range_str.split('-'))
                            display_range = (start, end)
                        else:
                            display_range = (int(range_str), int(range_str)) # Single line

                        if highlight_part:
                            try:
                                highlight_val = highlight_part.replace('highlight=', '').strip()
                                highlight_lines = [int(x.strip()) for x in highlight_val.split(',')]
                            except ValueError:
                                print(f"Warning: Could not parse highlight lines in: {line}")

                sections.append({
                    'type': marker_type,
                    'display_range': display_range, # (start, end) or 'ALL' or None for BLANK
                    'highlight_lines': highlight_lines,
                    'text': text
                })
            else:
                print(f"Warning: Skipping unparseable line in transcript: {line}")

    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_file_path}")
        return None, None
    except Exception as e:
        print(f"Error parsing transcript: {e}")
        return None, None

    return current_code_file, sections

# --- Main Automation Logic ---
def create_video_from_script():
    print(f"Parsing transcript from {TRANSCRIPT_FILE}...")
    code_file_path, sections = parse_transcript(TRANSCRIPT_FILE)

    if not code_file_path or not sections:
        print("Failed to parse transcript. Exiting.")
        return

    print(f"Using code file: {code_file_path}")
    try:
        with open(code_file_path, 'r', encoding='utf-8') as f:
            all_code_lines = [line.rstrip() for line in f.readlines()] # Remove trailing newlines
    except FileNotFoundError:
        print(f"Error: Code file not found at {code_file_path}. Exiting.")
        return

    print("Generating audio and image sequences...")
    audio_clips = []
    image_paths = []
    current_image_index = 0

    for i, section in enumerate(sections):
        print(f"  Processing section {i+1}/{len(sections)}: '{section['text'][:50]}...'")

        # 1. Generate Audio
        audio_text = section['text']
        audio_filename = os.path.join(TEMP_DIR, f'audio_segment_{i:04d}.mp3')
        try:
            tts = gTTS(text=audio_text, lang='en', slow=False)
            tts.save(audio_filename)
        except Exception as e:
            print(f"Error generating audio for section {i}: {e}. Skipping this section.")
            continue # Skip this section if audio generation fails

        audio_duration = get_audio_duration(audio_filename)
        audio_clips.append(AudioFileClip(audio_filename))

        # 2. Generate Corresponding Image
        if section['type'] == 'BLANK':
            # Create a blank image
            img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), color=BACKGROUND_COLOR)
            img_path = os.path.join(TEMP_DIR, f'code_frame_{current_image_index:04d}.png')
            img.save(img_path)
            image_paths.append(img_path)
            current_image_index += 1
        else: # LINE or ALL
            display_start, display_end = 0, len(all_code_lines) - 1
            if section['display_range'] != 'ALL':
                display_start = section['display_range'][0]
                display_end = section['display_range'][1]

            img_path = create_code_image(
                all_code_lines,
                display_start,
                display_end,
                section['highlight_lines'],
                current_image_index
            )
            image_paths.append(img_path)
            current_image_index += 1

    print("Assembling video...")
    if not image_paths or not audio_clips:
        print("No valid content generated for video. Exiting.")
        return

    # Create video clips from images, setting duration based on audio
    video_segments = []
    for i in range(len(image_paths)):
        try:
            image_clip = ImageSequenceClip([image_paths[i]], fps=FPS) # moviepy expects list of image paths
            image_clip = image_clip.with_duration(audio_clips[i].duration)
            video_segments.append(image_clip)
        except IndexError: # In case audio_clips and image_paths don't perfectly align (due to skipped audio)
            print(f"Warning: Mismatch between image {i} and audio clip. Skipping image.")
            continue

    if not video_segments:
        print("No video segments could be created. Exiting.")
        return

    # Concatenate video segments
    final_video_clip = concatenate_videoclips(video_segments, method="compose")
    final_audio_clip = concatenate_audioclips(audio_clips) # Concatenate audio

    final_video_clip = final_video_clip.with_audio(final_audio_clip)

    print(f"Writing final video to {OUTPUT_VIDEO_FILE}...")
    try:
        final_video_clip.write_videofile(OUTPUT_VIDEO_FILE, fps=FPS, codec="libx264")
        print("Video creation complete!")
    except Exception as e:
        print(f"Error writing video file: {e}")

    # Clean up temporary files
    # print("Cleaning up temporary files...")
    # for f in os.listdir(TEMP_DIR):
    #     os.remove(os.path.join(TEMP_DIR, f))
    # os.rmdir(TEMP_DIR)
    # print("Cleanup complete.")

if __name__ == '__main__':
    create_video_from_script()