package org.mg.wekalib.eval2.util;

import java.io.Serializable;

import org.mg.javalib.io.KeyValueFileStore;

public class ResultProvider
{
	static KeyValueFileStore<String, Serializable> STORE = new KeyValueFileStore<>("/tmp/store");

	public static boolean contains(String key)
	{
		return STORE.contains(key);
	}

	public static Serializable get(String key)
	{
		return STORE.get(key);
	}

	public static void set(String key, Serializable value)
	{
		STORE.store(key, value);
	}
}
