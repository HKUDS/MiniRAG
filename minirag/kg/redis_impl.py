import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import pipmaster as pm

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis
from minirag.utils import logger
from minirag.base import BaseKVStorage
import json
import copy
from minirag.utils import merge_tuples


@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get("REDIS_URI", "redis://localhost:6379")
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Use Redis as KV {self.namespace}")

    async def all_keys(self) -> list[str]:
        keys = await self._redis.keys(f"{self.namespace}:*")
        return [key.split(":", 1)[-1] for key in keys]

    async def get_by_id(self, id):
        data = await self._redis.get(f"{self.namespace}:{id}")
        return json.loads(data) if data else None

    async def get_by_ids(self, ids, fields=None):
        pipe = self._redis.pipeline()
        for id in ids:
            pipe.get(f"{self.namespace}:{id}")
        results = await pipe.execute()

        values = [
            json.loads(result) if result is not None else None for result in results
        ]
        if fields is not None:
            # Filter fields if specified, preserving missing IDs and empty projections.
            return [
                {field: value.get(field) for field in fields if field in value}
                if value is not None
                else None
                for value in values
            ]

        return values

    async def filter_keys(self, data: list[str]) -> set[str]:
        pipe = self._redis.pipeline()
        for key in data:
            pipe.exists(f"{self.namespace}:{key}")
        results = await pipe.execute()

        existing_ids = {data[i] for i, exists in enumerate(results) if exists}
        return set(data) - existing_ids

    async def upsert(self, data: dict[str, dict]):
        pipe = self._redis.pipeline()
        for k, v in tqdm_async(data.items(), desc="Upserting"):
            pipe.set(f"{self.namespace}:{k}", json.dumps(v))
        await pipe.execute()

        for k in data:
            data[k]["_id"] = k
        return data

    async def drop(self):
        keys = await self._redis.keys(f"{self.namespace}:*")
        if keys:
            await self._redis.delete(*keys)
