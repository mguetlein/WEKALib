package org.mg.wekalib.eval2.job;

import java.io.Serializable;

import org.mg.javalib.io.KeyValueFileStore;

public class ResultProvider
{
	KeyValueFileStore<String, Serializable> keyValueStore = new KeyValueFileStore<>("jobs/store");

	public boolean contains(String key)
	{
		return keyValueStore.contains(key);
	}

	public Serializable get(String key)
	{
		return keyValueStore.get(key);
	}

	public void set(String key, Serializable value)
	{
		keyValueStore.store(key, value);
	}
}
