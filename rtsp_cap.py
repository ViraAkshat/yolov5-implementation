import os
import concurrent.futures
import subprocess

def run_ffmpeg_in_screen(rtsp_url, output_file, screen_name):
    ffmpeg_cmd = [
        "ffmpeg",
	    "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vcodec", "copy",
        "-acodec", "copy",
        "-t", "60",  # Recording duration: 60 seconds (1 minute)
        os.path.abspath(output_file)  # Use the full path for the output file
    ]

    screen_cmd = ["screen", "-dmS", screen_name] + ffmpeg_cmd

    try:
        subprocess.run(screen_cmd, check=True)
        print("Recording for {} has started in a detached screen session.".format(rtsp_url))
    except subprocess.CalledProcessError as e:
        print("Error: {}".format(e))

# Example usage:
if __name__ == "__main__":
    rtsp_streams = [
        {"url": "rtsp", "output_file": "0101.mp4", "screen_name": "0101ch"},
        # Add more RTSP streams as needed
    ]

    output_folder = "output_videos"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_stream = {executor.submit(run_ffmpeg_in_screen, stream["url"], os.path.join(output_folder, stream["output_file"]), stream["screen_name"]): stream for stream in rtsp_streams}

        for future in concurrent.futures.as_completed(future_to_stream):
            stream = future_to_stream[future]
            try:
                future.result()
            except Exception as e:
                print("Error processing {}: {}".format(stream["url"], e))

    print("All FFmpeg processes started in detached screen sessions.")
