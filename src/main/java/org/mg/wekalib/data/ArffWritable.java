package org.mg.wekalib.data;

public interface ArffWritable
{
	public String getRelationName();

	public int getNumAttributes();

	public String getAttributeName(int attribute);

	/**
	 * numeric attributes should return null
	 * 
	 * @param attribute
	 * @return
	 */
	public String[] getAttributeDomain(int attribute);

	public int getNumInstances();

	/**
	 * nominal features: return index in domain
	 * missing values: return Double.NaN
	 * 
	 * @param instance
	 * @param attribute
	 * @return
	 * @throws Exception
	 */
	public double getAttributeValueAsDouble(int instance, int attribute) throws Exception;

	public boolean isSparse();
}
