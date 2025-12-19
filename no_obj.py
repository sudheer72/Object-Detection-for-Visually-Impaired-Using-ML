from gtts import gTTS

# Text for "No object detected"
text = "No object detected"

# Generate speech
tts = gTTS(text=text, lang="en")

# Save the output as an MP3 file
tts.save("no_obj.mp3")  # Save as 'no_obj.mp3' instead of 'output.mp3'

print("✅ 'no_obj.mp3' has been successfully generated!")
