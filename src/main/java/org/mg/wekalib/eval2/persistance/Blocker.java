package org.mg.wekalib.eval2.persistance;


public interface Blocker
{
	public boolean isBlockedByThread(String key, String threadId);

	public boolean block(String key, String threadId);

	public void unblock(String key);

	public void clear();
}