import asyncio

class AsyncCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = asyncio.Lock()

    async def increment(self, by=1):
        async with self._lock:
            self.value += by

    async def decrement(self, by=1):
        async with self._lock:
            self.value -= by

    async def get_value(self):
        async with self._lock:
            return self.value