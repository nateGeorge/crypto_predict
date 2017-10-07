# core
import os
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# installed
import telegram as t
import telegram.ext as te

CS_KEY = os.environ.get('cs_bot_token')

def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                    text=("Welcome, I'm the coin shark!\n"
                    "I'm here to help you boost up your cash."))


def help(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                    text=("Here are the commands you can use:\n\n"
                    "(times can be used in 12 or 24-hour format; for 12-hour use am and pm suffixes)\n\n"
                    "Notification settings:\n"
                    "/frequency 1h : sets the notification frequency to 1 hour.  "
                    "Any integer and h for hour and m for minute are available\n\n"
                    "/summary on at 8am : displays a morning summary at 8am.  "
                    "/summary off turns off the daily summary\n\n"
                    "/daily 8am : sets the notification frequency to once daily, at 8am.\n\n"
                    "/topn 10 : number of top results to show for predicted gainers and losers\n\n"
                    "/"))


def handle_msg(bot, update):
    msg = update.message.text
    bot.send_message(chat_id=update.message.chat_id, text="I can't talk yet")


def unknown(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Hmm, I'm not understanding what you're getting at.  Try /help")


if __name__ == "__main__":
    bot = t.Bot(token=CS_KEY)
    updater = te.Updater(token=CS_KEY)  # gets updates from chat
    dispatcher = updater.dispatcher  # sends updates to chats

    # make command handlers
    start_handler = te.CommandHandler('start', start)  # handles things like /start
    dispatcher.add_handler(start_handler)

    help_handler = te.CommandHandler('help', help)  # handles things like /start
    dispatcher.add_handler(help_handler)

    # make message handlers
    main_message_handler = te.MessageHandler(te.Filters.text, handle_msg)
    dispatcher.add_handler(main_message_handler)

    # must be added last, so other handlers can look at the message first
    unknown_handler = te.MessageHandler(te.Filters.command, unknown)
    dispatcher.add_handler(unknown_handler)

    updater.start_polling()  # starts the bot
