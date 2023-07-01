using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Stores boundary information for a hexagon (1 bit per edge)
[Serializable]
public class HexBoundary
{
    public byte edges;

    private static readonly int UP_RIGHT = 0b1;
    private static readonly int RIGHT = 0b10;
    private static readonly int DOWN_RIGHT = 0b100;
    private static readonly int DOWN_LEFT = 0b1000;
    private static readonly int LEFT = 0b10000;
    private static readonly int UP_LEFT = 0b100000;

    private static readonly Dictionary<HecsCoord, int> LOC_TO_EDGE = new Dictionary<HecsCoord, int>
    {
        {HecsCoord.ORIGIN.UpRight(), UP_RIGHT},
        {HecsCoord.ORIGIN.Right(), RIGHT},
        {HecsCoord.ORIGIN.DownRight(), DOWN_RIGHT},
        {HecsCoord.ORIGIN.DownLeft(), DOWN_LEFT},
        {HecsCoord.ORIGIN.Left(), LEFT},
        {HecsCoord.ORIGIN.UpLeft(), UP_LEFT},
    };

    public static HexBoundary FromBinary(byte e)
    {
        var b = new HexBoundary();
        b.Deserialize(e);
        return b;
    }

    public HexBoundary()
    {
        edges = 0;
    }

    public override string ToString()
    {
        return base.ToString() + ": edges: " + edges;
    }

    // Accessor methods.
    public bool UpRight() { return (edges & UP_RIGHT) != 0; }
    public bool Right() { return (edges & RIGHT) != 0; }
    public bool DownRight() { return (edges & DOWN_RIGHT) != 0; }
    public bool DownLeft() { return (edges & DOWN_LEFT) != 0; }
    public bool Left() { return (edges & LEFT) != 0; }
    public bool UpLeft() { return (edges & UP_LEFT) != 0; }

    // Accessor method. Takes in the cell location and a neighbor coordinate.
    // Gets the edge shared with that neighbor.
    //
    // TODO(sharf): It's inconvenient that a boundary doesn't know its own 
    // location, you shouldn't need to pass loc into this function.
    public bool GetEdgeWith(HecsCoord loc, HecsCoord neighbor)
    {
        HecsCoord displacement = HecsCoord.Sub(neighbor, loc);
        if (!LOC_TO_EDGE.ContainsKey(displacement)) return false;
        return (edges & LOC_TO_EDGE[displacement]) != 0;
    }

    // Mutator methods.
    public void UpRight(bool e) { SetBit(e, UP_RIGHT); }
    public void Right(bool e) { SetBit(e, RIGHT); }
    public void DownRight(bool e) { SetBit(e, DOWN_RIGHT); }
    public void DownLeft(bool e) { SetBit(e, DOWN_LEFT); }
    public void Left(bool e) { SetBit(e, LEFT); }
    public void UpLeft(bool e) { SetBit(e, UP_LEFT); }

    // Mutator method. Takes in the cell location and a neighbor coordinate.
    // Sets the edge shared with that neighbor.
    public void SetEdgeWith(HecsCoord loc, HecsCoord neighbor)
    {
        HecsCoord displacement = HecsCoord.Sub(neighbor, loc);
        if (!LOC_TO_EDGE.ContainsKey(displacement)) return;
        edges |= (byte)LOC_TO_EDGE[displacement];
    }

    // Similar to SetEdgeWith, but clears the corresponding edge.
    public void ClearEdgeWith(HecsCoord loc, HecsCoord neighbor)
    {
        HecsCoord displacement = HecsCoord.Sub(neighbor, loc);
        if (!LOC_TO_EDGE.ContainsKey(displacement)) return;
        edges &= ((byte)~LOC_TO_EDGE[displacement]);
    }

    public void AllBlocked()
    {
        edges = 0xFF;
    }

    public void Clear()
    {
        edges = 0;
    }

    // Save the current edge boundary info to a single byte.
    public byte Serialize() { return edges; }

    // Warning, this overwrites the current edge state.
    public void Deserialize(byte value) { edges = value; }

    public void MergeWith(HexBoundary other)
    {
        edges |= other.edges;
    }

    private void SetBit(bool val, int bit)
    {
        int binVal = (val) ? 1 : 0;
        int mask = 1 << bit;
        byte clearMask = (byte)(~mask);
        edges &= clearMask;
        edges |= (byte)(binVal << bit);
    }

    private int OppositeSide(int side)
    {
        side <<= 3;
        // Wraparound logic;
        if (side > UP_LEFT)
        {
            side >>= 6;
        }
        return side;
    }
}
