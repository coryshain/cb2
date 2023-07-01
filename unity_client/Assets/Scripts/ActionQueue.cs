﻿using System;
using System.Collections.Generic;
using UnityEngine;


// This class is a queue of actions that some object shall take. Custom actions 
// can be implemented via the IAction interface and each action can be
// associated with its own animation type.

// This is generally used to animate game objects, however it is also repurposed for UI animation.
public class ActionQueue
{
    // Enable to get action queue debug messages.
    [Serializable]
    public enum AnimationType
    {
        NONE,
        IDLE,
        WALKING,
        INSTANT,
        TRANSLATE,
        ACCEL_DECEL,
        SKIPPING,
        ROTATE,
        FADE,
    }

    public class ActionInfo
    {
        public AnimationType Type;
        public HecsCoord Displacement;  // Object displacement in HECS coords.
        public float Rotation;  // Heading in degrees, 0 = north, clockwise.
        public float BorderRadius;  // Radius of the object's outline, if applicable.
        public Network.Color BorderColor;  // Color of the object's outline, if applicable.
        public float Opacity;  // Used for UI element animations.
        public float DurationS;  // Duration in seconds.
        public DateTime Expiration;  // If the action delays past this deadline, fastforward to next action.
        public Network.Color BorderColorFollowerPov;  // Color of the object's outline, if applicable.
    };

    // A kind of action (walk, skip, jump). This is distinctly different from 
    // the model animation (though related).  The action defines how an object
    // moves through space as a function of time. The animation defines how the
    // object mesh changes (leg action motion, etc) as the object moves.
    public interface IAction
    {
        // Calculate intermediate state, given initial conditions and progress.
        // 
        // progress represents the action's completion (1.0 = completely done).
        State.Continuous Interpolate(State.Discrete initialConditions,
                                     float progress);
        // Calculate the next state, given the current state and an action.
        State.Discrete Transfer(State.Discrete s);
        // Action's duration in seconds.
        float DurationS();
        // Action's expiration date in system time.
        DateTime Expiration();
        // Convert this action to a packet that can be sent over the network.
        Network.Action Packet(int id);
    }

    private Queue<IAction> _actionQueue;
    private State.Discrete _state;
    private State.Discrete _targetState;
    private bool _actionInProgress;
    private DateTime _actionStarted;
    private float _progress;  // Progress of the current action. 0 -> 1.0f.
    private HexGrid _grid;
    private string _name;

    private System.Object _animationLock = new System.Object();

    private Logger _logger;

    public ActionQueue(string name="")
    {
        _actionQueue = new Queue<IAction>();
        _actionInProgress = false;
        _actionStarted = DateTime.UtcNow;
        _state = new State.Discrete();
        _targetState = new State.Discrete();
        _name = name;
        _logger = Logger.GetOrCreateTrackedLogger(name);
    }

    // Adds a move to the queue.
    public void AddAction(IAction action)
    {
        lock(_animationLock) {
            if (action == null)
            {
                return;
            }
            _targetState = action.Transfer(_targetState);
            _actionQueue.Enqueue(action);
            _logger.Debug("Enqueued action duration: " + action.DurationS());
            _logger.Debug("Enqueued action expiration: " + action.Expiration());
        }
    }

    // Is the object in the middle of a action.
    public bool IsBusy()
    {
        lock(_animationLock) {
            return _actionInProgress;
        }
    }

    public int PendingActions()
    {
        lock(_animationLock) {
            return _actionQueue.Count;
        }
    }

    public State.Continuous ContinuousState()
    {
        lock(_animationLock) {
            if (IsBusy())
            {
                return _actionQueue.Peek().Interpolate(_state, _progress);
            }
            return _state.Continuous();
        }
    }

    public State.Discrete State()
    {
        lock(_animationLock) {
            return _state;
        }
    }

    public State.Discrete TargetState()
    {
        lock(_animationLock) {
            return _targetState;
        }
    }

    // Wipe all current actions.
    public void Flush()
    {
        lock(_animationLock) {
            _actionQueue.Clear();
            _actionInProgress = false;
            _progress = 0.0f;
            _targetState = _state;
        }
    }

    public void Update()
    {
        lock(_animationLock) {
            // If there's no animation in progress, begin the next animation in the queue.
            if ((_actionQueue.Count > 0) && !_actionInProgress)
            {
                _logger.Debug(_name + " Q Start: " + _actionQueue.Peek());
                _progress = 0.0f;
                _actionStarted = DateTime.UtcNow;
                _actionInProgress = true;
                return;
            }

            TimeSpan delta = DateTime.UtcNow - _actionStarted;

            // Immediately skip any expired animations. Once triggered, this will
            // continue to fast-forward until the queue is empty or an unexpired
            if (_actionInProgress && (DateTime.UtcNow > _actionQueue.Peek().Expiration()))
            {
                while ((_actionQueue.Count > 0) && (DateTime.UtcNow > _actionQueue.Peek().Expiration()))
                {
                    _logger.Debug(_name + " Q Fast-forwarding expired action of duration " 
                                    + _actionQueue.Peek().DurationS() + "s. Duration: "
                                    + delta.Seconds);
                    _state = _actionQueue.Peek().Transfer(_state);
                    _actionQueue.Dequeue();
                    _actionInProgress = false;
                    return;
                }
            }

            // Convert to milliseconds for higher-resolution progress.
            if (_actionInProgress)
            {
                _progress = ((float)delta.TotalMilliseconds) /
                            (_actionQueue.Peek().DurationS() * 1000.0f);
            }

            if (_actionInProgress &&
                (delta.TotalMilliseconds > (_actionQueue.Peek().DurationS() * 1000.0f)))
            {
                _logger.Debug(_name + " Q Finish: " + _actionQueue.Peek());
                _state = _actionQueue.Peek().Transfer(_state);
                _actionQueue.Dequeue();
                _actionInProgress = false;
                return;
            }
        }
    }

}
