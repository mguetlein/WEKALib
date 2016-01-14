package org.mg.wekalib.eval2.persistance;

import java.lang.management.ManagementFactory;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.ThreadUtil;
import org.mg.wekalib.eval2.job.Printer;

public class BlockerImpl implements Blocker
{
	private KeyValueFileStore<String, String> keyValueStore;

	public BlockerImpl(String dir)
	{
		keyValueStore = new KeyValueFileStore<>(dir, false, false, null, false);
	}

	@Override
	public void clear()
	{
		keyValueStore.clear();
	}

	@Override
	public boolean isBlockedByThread(String key, String threadId)
	{
		return keyValueStore.contains(key) && threadId.equals(keyValueStore.get(key));
	}

	@Override
	public boolean block(String key, String threadId)
	{
		try
		{
			if (keyValueStore.contains(key))
			{
				Printer.println("already blocked: " + key);
				return false;
			}
			//			System.err.println("blocking: " + key);
			keyValueStore.store(key, threadId);
			ThreadUtil.sleep(1000);
			boolean blockSucceeded = threadId.equals(keyValueStore.get(key));
			if (!blockSucceeded)
				Printer.println_copyToError("could not block: " + key + " NOT EQUAL: " + threadId
						+ " != " + keyValueStore.get(key));
			return blockSucceeded;
		}
		catch (Exception e)
		{
			Printer.println_copyToError(
					"could not block: " + key + " because of " + e.getMessage());
			return false;
		}
	}

	@Override
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
