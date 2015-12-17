package org.mg.wekalib.eval2;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.ThreadUtil;

public class Blocker
{
	static KeyValueFileStore<Integer, Long> STORE = new KeyValueFileStore<>("/tmp/block");
	static
	{
		STORE.clear();
	}

	public static boolean block(int hashKey)
	{
		if (STORE.contains(hashKey))
			return false;
		STORE.store(hashKey, Thread.currentThread().getId());
		ThreadUtil.sleep(100);
		long threadId = STORE.get(hashKey);
		return Thread.currentThread().getId() == threadId;
	}

	public static void unblock(int hashKey)
	{
		if (!STORE.contains(hashKey))
			throw new IllegalStateException();
		STORE.clear(hashKey);
	}
}
