import { useState } from "react"
import { z } from "zod"
import { Input } from "./ui/input"
import { Button } from "./ui/button"
import { callBackend } from "../api/callBackend"


const MessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  text: z.string()
})

const ChatPropsSchema = z.object({
  api: z.string()
})

const BackendResponseSchema = z.object({
  result: z.string()
})

type Message = z.infer<typeof MessageSchema>
type ChatProps = z.infer<typeof ChatPropsSchema>

export function Chat({ api }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState<string>("")
  const [error, setError] = useState<string>("")

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()

    if (!input.trim()) return

    const userMessage: Message = { role: "user", text: input }
    
    // Validate message before adding
    try {
      MessageSchema.parse(userMessage)
    } catch (err) {
      setError("Invalid message format")
      return
    }

    // Add user message
    setMessages(prev => [...prev, userMessage])

    const currentInput = input
    // Empty input
    setInput("")
    setError("")

    try {
      const result = await callBackend(api, {
        query: currentInput
      })

      // Validate backend response
      //const validatedResult = BackendResponseSchema.parse(result)

      // Add response
      const assistantMessage: Message = { 
        role: "assistant", 
        text: result.response
      }
      
      MessageSchema.parse(assistantMessage)
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      console.error(err)
      if (err instanceof z.ZodError) {
        setError("Invalid response format from server")
      } else {
        setError("Failed to get response")
      }
    }
  }

  return (
    <div className="w-full max-w-lg mx-auto space-y-4">
      {/* Chat Window */}
      <section className="border h-96 p-4 overflow-y-auto rounded-lg bg-white shadow">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`mb-2 p-2 rounded-lg ${
              m.role === "user"
                ? "bg-blue-100 text-right"
                : "bg-gray-100 text-left"
            }`}
          >
            {m.text}
          </div>
        ))}
        
        {error && (
          <div className="mb-2 p-2 rounded-lg bg-red-100 text-red-700">
            {error}
          </div>
        )}
      </section>

      <section>
        <form className="flex gap-2" onSubmit={handleSubmit}>
          <Input
            placeholder="Ask something"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <Button type="submit">Send</Button>
        </form>
      </section>
    </div>
  )
}