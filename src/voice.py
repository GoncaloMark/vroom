import speech_recognition as sr
import pynng
import time
import json

class VoiceRecognition:
    def __init__(self) -> None:
        """
        Initializes the VoiceRecognition class, setting up the NanoMsg publisher,
        speech recognizer, and calibrating the microphone for ambient noise.
        
        The microphone is calibrated to adjust for background noise before starting.

        It listens for specific commands and publishes them using the NanoMsg pub-sub pattern.

        Attributes:
            publisher (pynng.Pub0): The NanoMsg publisher for sending commands.
            recognizer (sr.Recognizer): The speech recognition instance.
            microphone (sr.Microphone): The microphone input for audio capture.
            is_running (bool): Flag indicating if the process is still running.
        """
        self.publisher = pynng.Pub0(listen="tcp://127.0.0.1:5555")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_running = True
        
        # Adjust for ambient noise
        with self.microphone as source:
            print("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Calibration complete!")

    def process_audio(self) -> None:
        """
        Continuously listens to the microphone for audio input and processes
        recognized speech commands. The method publishes the commands via NanoMsg.
        
        Commands include color commands ("red", "blue", "green"), and actions
        ("delete", "erase", "clear", "stop"). If "stop" is detected, it stops
        the listening process.
        
        This method loops continuously until the `is_running` flag is set to False.
        """
        while self.is_running:
            with self.microphone as source:
                print("Listening...")
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    print("Processing speech...")
                    
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {text}")
                    
                    command = None
                    
                    if "red" in text:
                        command = 'red'
                    elif "blue" in text:
                        command = 'blue'
                    elif "green" in text:
                        command = 'green'
                    elif "delete" in text or "erase" in text:
                        command = 'delete'
                    elif "clear" in text:
                        command = 'clear'
                    elif "stop" in text:
                        command = 'stop'
                        self.is_running = False
                    
                    # Publish command if found
                    if command:
                        message = json.dumps({"command": command, "timestamp": time.time()})
                        self.publisher.send(message.encode())
                        print(f"Published command: {command}")
                        
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                
                time.sleep(0.1)  # Small delay to prevent CPU overuse

    def run(self) -> None:
        """
        Starts the voice recognition process by calling the `process_audio` method.
        This method serves as the entry point for starting the voice recognition loop.
        """
        print("Starting voice recognition...")
        try:
            self.process_audio()
        finally:
            self.publisher.close()

if __name__ == "__main__":
    voice_recognition = VoiceRecognition()
    voice_recognition.run()