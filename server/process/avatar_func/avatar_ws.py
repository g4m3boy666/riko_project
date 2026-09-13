import asyncio
import json
import threading

import websockets


HOST = "127.0.0.1"
PORT = 8765

_loop = None
_thread = None
_clients = set()
_started = threading.Event()


async def _handler(websocket, *args):
    print("[AVATAR] Godot connected")

    _clients.add(websocket)

    try:
        async for message in websocket:
            print(f"[GODOT] {message}")

    except websockets.ConnectionClosed:
        pass

    finally:
        _clients.discard(websocket)
        print("[AVATAR] Godot disconnected")


async def _broadcast(data):
    if not _clients:
        return

    message = json.dumps(data)

    dead_clients = []

    for client in list(_clients):
        try:
            await client.send(message)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        _clients.discard(client)


async def _server_main():
    global _loop

    _loop = asyncio.get_running_loop()

    async with websockets.serve(
        _handler,
        HOST,
        PORT,
    ):
        print(
            f"[AVATAR] WebSocket server running "
            f"on ws://{HOST}:{PORT}"
        )

        _started.set()

        await asyncio.Future()


def _thread_main():
    asyncio.run(_server_main())


def start_avatar_server():
    global _thread

    if _thread is not None and _thread.is_alive():
        return

    _thread = threading.Thread(
        target=_thread_main,
        daemon=True,
        name="RikoAvatarWebSocket",
    )

    _thread.start()

    _started.wait(timeout=5)


def send_avatar(data):
    if _loop is None:
        return

    asyncio.run_coroutine_threadsafe(
        _broadcast(data),
        _loop,
    )


def set_state(state):
    send_avatar(
        {
            "type": "state",
            "value": state,
        }
    )


def set_emotion(emotion):
    send_avatar(
        {
            "type": "emotion",
            "value": emotion,
        }
    )


def look(direction):
    send_avatar(
        {
            "type": "look",
            "value": direction,
        }
    )


def blink():
    send_avatar(
        {
            "type": "blink",
        }
    )


def reset_face():
    send_avatar(
        {
            "type": "reset",
        }
    )


def set_viseme(viseme):
    send_avatar(
        {
            "type": "viseme",
            "value": viseme,
        }
    )
