import { OllamaFunctions } from "langchain/experimental/chat_models/ollama_functions";
import { HumanMessage } from "langchain/schema";

import createBookingIfAvailable from "../tools/createBooking";
import deleteBooking from "../tools/deleteBooking";
import getAvailability from "../tools/getAvailability";
import getBookings from "../tools/getBookings";
import sendBookingEmail from "../tools/sendBookingEmail";
import updateBooking from "../tools/updateBooking";
import type { User, UserList } from "../types/user";

/**
 * Core of the Cal.ai booking agent: a LangChain Agent Executor.
 * Uses a toolchain to book meetings, list available slots, etc.
 * Uses OpenAI functions to better enforce JSON-parsable output from the LLM.
 */
const agent = async (
  input: string,
  user: User,
  users: UserList,
  apiKey: string,
  userId: number,
  agentEmail: string
) => {
  console.log(input, user, users, apiKey, userId, agentEmail);
  const tools = [
    // getEventTypes(apiKey),
    getAvailability(apiKey),
    getBookings(apiKey, userId),
    createBookingIfAvailable(apiKey, userId, users),
    updateBooking(apiKey, userId),
    deleteBooking(apiKey),
    sendBookingEmail(apiKey, user, users, agentEmail),
  ];

  const toolSystemPromptTemplate = `You are Cal.ai - a bleeding edge scheduling assistant that interfaces via email.
  Make sure your final answers are definitive, complete and well formatted.
  You have access to the following tools, you can ONLY USE THE ONES DEFINED HERE:

{
    getAvailability(${apiKey}),
    getBookings(${apiKey}, ${userId}),
    createBookingIfAvailable(${apiKey}, ${userId}, ${users}),
    updateBooking(${apiKey}, ${userId}),
    deleteBooking(${apiKey}),
    sendBookingEmail(${apiKey}, ${user}, ${users}, ${agentEmail}),

}

To use a tool, respond with a JSON object with the following structure:
{{
  "tool": <name of the called tool>,
  "tool_input": <parameters for the tool matching the above JSON schema>
}}

  Sometimes, tools return errors. In this case, try to handle the error intelligently or ask the user for more information.
  Tools will always handle times in UTC, but times sent to users should be formatted per that user's timezone.
  In responses to users, always summarize necessary context and open the door to follow ups. For example "I have booked your chat with @username for 3pm on Wednesday, December 20th, 2023 EST. Please let me know if you need to reschedule."
  If you can't find a referenced user, ask the user for their email or @username. Make sure to specify that usernames require the @username format. Users don't know other users' userIds.

  The primary user's id is: 18
  The primary user's username is: random
  The current time in the primary user's timezone is: 3:00PM
  The primary user's time zone is: EST
  
     The email references the following @usernames and emails: 
      id: 1, username: @onboarding, email: onboarding@gmail.com`;

  const model = new OllamaFunctions({
    temperature: 0,
    model: "mistral",
    toolSystemPromptTemplate,
  }).bind({
    functions: [
      {
        name: "getAvailability",
        description: "Get the times that the user is available",
        parameters: {
          type: "object",
          properties: {
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: [""],
        },
      },
      {
        name: "getBookings",
        description: "Retrieve a list of all bookings",
        parameters: {
          type: "object",
          properties: {
            startDate: {
              type: "string",
              format: "date-time",
              description: "The start date for the range of bookings to retrieve",
            },
            endDate: {
              type: "string",
              format: "date-time",
              description: "The end date for the range of bookings to retrieve",
            },
          },
          required: ["startDate", "endDate"],
        },
      },

      {
        name: "createBookingIfAvailable",
        description: "Create a booking if the time slot is available",
        parameters: {
          type: "object",
          properties: {
            dateTime: {
              type: "string",
              format: "date-time",
              description: "The date and time for the booking",
            },
            customerDetails: {
              type: "object",
              properties: {
                name: { type: "string" },
                email: { type: "string", format: "email" },
              },
              required: ["name", "email"],
            },
          },
          required: ["dateTime", "customerDetails"],
        },
      },

      {
        name: "updateBooking",
        description: "Update the details of an existing booking",
        parameters: {
          type: "object",
          properties: {
            bookingId: {
              type: "string",
              description: "The unique identifier for the booking",
            },
            updatedDetails: {
              type: "object",
              properties: {
                dateTime: { type: "string", format: "date-time" },
                customerDetails: {
                  type: "object",
                  properties: {
                    name: { type: "string" },
                    email: { type: "string", format: "email" },
                  },
                },
              },
            },
          },
          required: ["bookingId", "updatedDetails"],
        },
      },

      {
        name: "deleteBooking",
        description: "Delete an existing booking",
        parameters: {
          type: "object",
          properties: {
            bookingId: {
              type: "string",
              description: "The unique identifier for the booking to delete",
            },
          },
          required: ["bookingId"],
        },
      },

      {
        name: "sendBookingEmail",
        description: "Send an email confirmation for a booking",
        parameters: {
          type: "object",
          properties: {
            bookingId: {
              type: "string",
              description: "The unique identifier for the booking",
            },
            email: {
              type: "string",
              format: "email",
              description: "Email address to send the confirmation to",
            },
          },
          required: ["bookingId", "email"],
        },
      },
    ],
    function_call: {
      name: "getAvailability",
      name: "getBookings",
      name: "createBookingIfAvailable",
      name: "updateBooking",
      name: "deleteBooking",
      name: "sendBookingEmail",
    },

    // You can set the `function_call` arg to force the model to use a function
    // function_call: {
    //   name: "get_current_weather",
    // },
  });

  /**
   * Initialize the agent executor with arguments.
   */
  //   const executor = await initializeAgentExecutorWithOptions(tools, model, {
  //     agentArgs: {
  //       prefix: `You are Cal.ai - a bleeding edge scheduling assistant that interfaces via email.
  // Make sure your final answers are definitive, complete and well formatted.
  // Sometimes, tools return errors. In this case, try to handle the error intelligently or ask the user for more information.
  // Tools will always handle times in UTC, but times sent to users should be formatted per that user's timezone.
  // In responses to users, always summarize necessary context and open the door to follow ups. For example "I have booked your chat with @username for 3pm on Wednesday, December 20th, 2023 EST. Please let me know if you need to reschedule."
  // If you can't find a referenced user, ask the user for their email or @username. Make sure to specify that usernames require the @username format. Users don't know other users' userIds.

  // The primary user's id is: ${userId}
  // The primary user's username is: ${user.username}
  // The current time in the primary user's timezone is: ${now(user.timeZone)}
  // The primary user's time zone is: ${user.timeZone}
  // The primary user's event types are: ${user.eventTypes
  //         .map((e: EventType) => `ID: ${e.id}, Slug: ${e.slug}, Title: ${e.title}, Length: ${e.length};`)
  //         .join("\n")}
  // The primary user's working hours are: ${user.workingHours
  //         .map(
  //           (w: WorkingHours) =>
  //             `Days: ${w.days.join(", ")}, Start Time (minutes in UTC): ${
  //               w.startTime
  //             }, End Time (minutes in UTC): ${w.endTime};`
  //         )
  //         .join("\n")}
  // ${
  //   users.length
  //     ? `The email references the following @usernames and emails: ${users
  //         .map(
  //           (u) =>
  //             `${
  //               (u.id ? `, id: ${u.id}` : "id: (non user)") +
  //               (u.username
  //                 ? u.type === "fromUsername"
  //                   ? `, username: @${u.username}`
  //                   : ", username: REDACTED"
  //                 : ", (no username)") +
  //               (u.email
  //                 ? u.type === "fromEmail"
  //                   ? `, email: ${u.email}`
  //                   : ", email: REDACTED"
  //                 : ", (no email)")
  //             };`
  //         )
  //         .join("\n")}`
  //     : ""
  // }
  //             `,
  //     },
  //     agentType: "openai-functions",
  //     returnIntermediateSteps: env.NODE_ENV === "development",
  //     verbose: env.NODE_ENV === "development",
  //   });

  const result = await model.invoke([
    new HumanMessage({
      content: input,
    }),
  ]);
  const { output } = result;

  return output;
};

export default agent;
