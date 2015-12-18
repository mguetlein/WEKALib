package org.mg.wekalib.eval2.util;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.ThreadUtil;

public class Blocker
{
	static KeyValueFileStore<String, Long> STORE = new KeyValueFileStore<>("/tmp/block");
	static
	{
		STORE.clear();
	}

	public static boolean block(String key)
	{
		if (STORE.contains(key))
			return false;
		STORE.store(key, Thread.currentThread().getId());
		ThreadUtil.sleep(100);
		long threadId = STORE.get(key);
		return Thread.currentThread().getId() == threadId;
	}

	public static void unblock(String key)
	{
		if (!STORE.contains(key))
			throw new IllegalStateException();
		STORE.clear(key);
	}
}
