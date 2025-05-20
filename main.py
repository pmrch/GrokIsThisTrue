from src.bot import Bot

bot = Bot()
bot.run()

'''async def main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    
    # Offload audio recording to a thread so it doesn't
    # block the event loop
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor as executor:
        loop.run_in_executor(executor = executor)#, record_audio)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())'''