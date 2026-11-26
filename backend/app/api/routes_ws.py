"""WebSocket endpoint streaming real-time simulation timestep snapshots."""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.api.deps import get_sim_service

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket):
    await websocket.accept()
    sim = get_sim_service()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    sim.register_ws_subscriber(loop, queue)

    try:
        # Push initial snapshot cleanly on connect
        initial_state = sim.get_state(include_history=True)
        await websocket.send_json({
            "type": "init",
            "state": initial_state.model_dump(),
        })

        async def _receiver():
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})

        recv_task = asyncio.create_task(_receiver())

        while True:
            get_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {recv_task, get_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if recv_task in done:
                get_task.cancel()
                # Propagate disconnect or exception
                recv_task.result()
            if get_task in done:
                payload = get_task.result()
                await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        sim.unregister_ws_subscriber(loop, queue)
