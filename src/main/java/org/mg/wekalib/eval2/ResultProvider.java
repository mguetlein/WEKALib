package org.mg.wekalib.eval2;

import java.io.Serializable;

import org.mg.javalib.io.KeyValueFileStore;

public class ResultProvider
{
	static KeyValueFileStore<Integer, Serializable> STORE = new KeyValueFileStore<>("/tmp/store");

	public static boolean contains(int hashKey)
	{
		return STORE.contains(hashKey);
	}

	public static Serializable get(int hashKey)
	{
		return STORE.get(hashKey);
	}

	public static void set(int hashKey, Serializable value)
	{
		STORE.store(hashKey, value);
	}
}
