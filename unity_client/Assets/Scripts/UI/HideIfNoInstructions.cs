using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HideIfNoInstructions : MonoBehaviour
{
    void Update()
    {
        List<Network.ObjectiveMessage> messages = MenuTransitionHandler.TaggedInstance().ObjectiveList();
        foreach (Network.ObjectiveMessage message in messages)
        {
            if (!message.is_concluded())
            {
                gameObject.transform.localScale = new Vector3(1, 1, 1);
                return;
            }
        }
        gameObject.transform.localScale = new Vector3(0, 0, 0);
    }
}