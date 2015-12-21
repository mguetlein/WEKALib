package org.mg.wekalib.eval2.job;

import java.lang.management.ManagementFactory;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.ThreadUtil;

public class Blocker
{
	private KeyValueFileStore<String, String> keyValueStore = new KeyValueFileStore<>("jobs/block");
	private static ThreadLocal<String> threadIDs = new ThreadLocal<>();

	public Blocker()
	{
		//		keyValueStore.clear();
	}

	static String getThreadID()
	{
		if (threadIDs.get() == null)
			threadIDs.set(ManagementFactory.getRuntimeMXBean().getName() + "-" + Thread.currentThread().getId());
		return threadIDs.get();
	}

	public boolean block(String key)
	{
		if (keyValueStore.contains(key))
			return false;
		keyValueStore.store(key, getThreadID());
		ThreadUtil.sleep(100);
		return getThreadID().equals(keyValueStore.get(key));
	}

	public void unblock(String key)
	{
		if (!keyValueStore.contains(key))
			throw new IllegalStateException();
		keyValueStore.clear(key);
	}

	public static void main(String[] args)
	{
		System.out.println(new String[] { "" });
		System.out.println(Thread.currentThread().hashCode());
		System.out.println(ManagementFactory.getRuntimeMXBean().getName());
	}
}
