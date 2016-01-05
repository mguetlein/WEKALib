package org.mg.wekalib.eval2.persistance;

import java.io.Serializable;

public interface ResultProvider
{
	public boolean contains(String key);

	public Serializable get(String key);

	public void set(String key, Serializable value);

	public void clear();
}