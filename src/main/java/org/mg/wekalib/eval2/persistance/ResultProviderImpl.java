package org.mg.wekalib.eval2.persistance;

import java.io.Serializable;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.ThreadUtil;

public class ResultProviderImpl implements ResultProvider
{
	KeyValueFileStore<String, Serializable> keyValueStore = new KeyValueFileStore<>("jobs/store",
			false, true, "jobs/tmp", true);

	@Override
	public boolean contains(String key)
	{
		return keyValueStore.contains(key);
	}

	@Override
	public Serializable get(String key)
	{
		try
		{
			return keyValueStore.get(key);
		}
		catch (Exception e)
		{
			ThreadUtil.sleep(1000);
			System.err.println("try getting result a second time: " + key);
			return keyValueStore.get(key);
		}
	}

	@Override
	public void set(String key, Serializable value)
	{
		keyValueStore.store(key, value);
	}

	@Override
	public void clear()
	{
		keyValueStore.clear();
	}
}
