from youtube_transcript_api import YouTubeTranscriptApi

def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    return [
        {"text": s.text, "start": s.start, "duration": s.duration}
        for s in transcript
    ]
    
    
