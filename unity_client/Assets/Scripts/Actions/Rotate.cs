﻿using System;
using UnityEngine;

public class Rotate : ActionQueue.IAction
{
    public static Rotate Turn(float rotation, float durationS)
    {
        return new Rotate(
            new ActionQueue.ActionInfo()
            {
                Type = ActionQueue.AnimationType.ROTATE,
                Displacement = HecsCoord.ORIGIN,
                Rotation = rotation,
                DurationS = durationS,
                Expiration = DateTime.UtcNow.AddSeconds(10),
            }
        );
    }

    private ActionQueue.ActionInfo _info;

    public Rotate(ActionQueue.ActionInfo info)
    {
        _info = info;
    }
    
    public float DurationS() { return _info.DurationS;  }
    public DateTime Expiration() { return _info.Expiration;  }

    public State.Continuous Interpolate(State.Discrete initialConditions, float progress)
    {
        // Cap progress at 1.0f.
        if (progress > 1.0f) progress = 1.0f;

        State.Discrete end = Transfer(initialConditions);
        
        State.Continuous interp = new State.Continuous();
        interp.Position = initialConditions.Vector();
        interp.HeadingDegrees = initialConditions.HeadingDegrees
	                            + _info.Rotation * progress; 
        interp.BorderRadius = initialConditions.BorderRadius;
        interp.Opacity = initialConditions.Opacity;
        interp.Animation = _info.Type;
        return interp; 
    }

    public State.Discrete Transfer(State.Discrete s)
    {
        s.HeadingDegrees += _info.Rotation;
        return s;
    }

    public Network.Action Packet(int id)
    {
        return new Network.Action()
        {
            id = id,
            action_type = Network.ActionType.ROTATE,
            animation_type = (Network.AnimationType)_info.Type,
            displacement = HecsCoord.ORIGIN,
            rotation = _info.Rotation,
            duration_s = _info.DurationS,
            expiration = _info.Expiration.ToString("o"),
        };
    }
}

