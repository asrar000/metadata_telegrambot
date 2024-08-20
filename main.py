import logging
import os
import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import pipeline
from keybert import KeyBERT
from datetime import datetime
 
nest_asyncio.apply()
# Initialize Logging
logging.basicConfig(
    filename='./bot.log',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Hugging Face Model Initialization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
title_generator = pipeline("summarization", model="google/pegasus-xsum")
keyword_extractor = KeyBERT()

# Conversational Model for Chatting
chat_model = pipeline("text-generation", model="distilgpt2")  # You can use any conversational model here

# Telegram Bot Commands
async def start(update: Update, context):
    await update.message.reply_text("Hello! I'm a conversational and metadata generator bot. Send me any text, and I'll chat with you and generate metadata based on your input!\nMade with 💖 by Asrar")

async def help_command(update: Update, context):
    await update.message.reply_text("To use this bot, simply send a message with some text or a question. I'll reply to you and generate metadata based on your input.")

# Handle User Input and Metadata Generation
async def handle_message(update: Update, context):
    user_input = update.message.text
    user = update.message.from_user.first_name

    # Log the received message with a timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Received message from {user} at {current_time}: {user_input}")
    await update.message.reply_text("Generating response and metadata...")

    # Generate a conversational response using GPT-Neo or any conversational model
    conversation_response = chat_model(user_input, max_length=500, do_sample=True)[0]['generated_text']

    # Respond to the user's query with a dynamic conversational reply
    response = f"Response:\n{conversation_response}"
    await update.message.reply_text(response)

    # Generate Metadata
    # 1. Summarization
    summary = summarizer(user_input, max_length=500, min_length=5, do_sample=False)[0]['summary_text']

    # 2. Title Generation (Simple text generation)
    title = title_generator(user_input, max_new_tokens=10, num_return_sequences=1)[0]['summary_text'].strip()

    # 3. Keyword/Tag Extraction
    keywords = keyword_extractor.extract_keywords(user_input, keyphrase_ngram_range=(1, 2), stop_words='english')
    tags = [word[0] for word in keywords]
    await update.message.reply_text('Metadata:')
    # Respond with Metadata
    metadata_response = (
        f"**Title**: {title}\n"
        f"**Summary**: {summary}\n"
        f"**Tags**: {', '.join(tags)}"
    )

    await update.message.reply_text(metadata_response)




if __name__ == "__main__":
    application = ApplicationBuilder().token(os.environ.get("Token")).build()
    print("The bot has started")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()