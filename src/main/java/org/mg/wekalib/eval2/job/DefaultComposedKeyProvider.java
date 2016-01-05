package org.mg.wekalib.eval2.job;

import java.io.File;

import org.apache.commons.codec.digest.DigestUtils;
import org.mg.javalib.util.ArrayUtil;

public abstract class DefaultComposedKeyProvider implements KeyProvider, ComposedKeyProvider
{
	protected String getKeyContent(Object... elements)
	{
		StringBuffer b = new StringBuffer();
		b.append(this.getClass().getSimpleName());
		for (Object o : ArrayUtil.flatten(elements))
		{
			b.append('#');
			if (o == null)
				b.append("null");
			else if (o instanceof ComposedKeyProvider)
				b.append(((ComposedKeyProvider) o).getKeyContent());
			else if (o instanceof Enum<?> || o instanceof String || o instanceof Integer
					|| o instanceof Double || o instanceof Long || o instanceof Boolean)
				b.append(o.toString());
			else
				throw new IllegalArgumentException("Not a key provider: " + o + " " + o.getClass());
		}
		return b.toString();
	}

	@Override
	public final String getKey()
	{
		String prefix = getKeyPrefix();
		return (prefix != null ? (prefix) : "") + File.separator
				+ DigestUtils.md5Hex(getKeyContent());
	}

}
